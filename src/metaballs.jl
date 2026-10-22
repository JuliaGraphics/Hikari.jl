# Metaballs: a surface traced straight from particles, the way a particle fluid's water
# is drawn without meshing it.
#
# The field is Blinn's blobby sum, here with the compact polynomial of Wyvill's soft
# objects: every particle adds `(1 - d²/R²)³` within its radius `R`, and the surface is
# where the sum is `iso`. The particles are kept sorted into a grid of cells `R` wide,
# so the field at a point is a sum over the 27 cells round it. Each cell is classed once
# per particle set by bounds on the field within it (see `classify_kernel!`): surely
# outside the surface, surely inside, or possibly crossed by it. A ray walks the cells
# (Amanatides & Woo) and samples the field only where it can cross; a sign change between
# two samples is refined by bisection, and the normal is the field's analytic gradient.

using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA

const METABALL_OUTSIDE = 0x00
const METABALL_SURFACE = 0x01
const METABALL_INSIDE = 0x02

"The integral of Wyvill's kernel `(1 - d²/R²)³` over space, over `R³`: `4π · 16/315`."
const WYVILL_VOLUME = 4f0 * Float32(π) * 16f0 / 315f0

"""
    Metaballs(points, radius; iso, step = radius / 6, anisotropy = nothing, chunk = 32768, backend)

The surface at level `iso` of the metaball field of `points`, each contributing
`(1 - d²/radius²)³` within `radius`. `step` is the spacing of the field samples a ray
takes where the surface may be; features thinner than it can be missed. The particles
are sorted into cells on the host and the cells classified on `backend`, `chunk` cells
a launch. With `anisotropy = Anisotropic()` every particle's kernel is an ellipsoid
shaped by its neighbours instead of a ball (see [`Anisotropic`](@ref)).
"""
struct Metaballs{P<:AbstractVector{Point3f},I<:AbstractVector{Int32},C<:AbstractVector{UInt8},S}
    points::P            # sorted by cell
    cellstart::I         # cell c's points are points[cellstart[c]:cellstart[c+1]-1]
    class::C             # METABALL_OUTSIDE / _SURFACE / _INSIDE per cell
    lo::Point3f          # the grid's lower corner
    dims::Vec{3,Int32}   # cells along x, y, z
    radius::Float32      # a particle's reach, and the cells' size
    iso::Float32
    step::Float32
    shapes::S            # `nothing` (balls) or the particles' `KernelShapes`, in `points`' order
end

function Metaballs(points::AbstractVector{<:Point3}, radius::Real; iso::Real, step::Real = radius / 6,
                   anisotropy = nothing, chunk = 32768, backend = KA.CPU())
    R = Float32(radius)
    pts = Point3f.(points)
    lo = Point3f(minimum(p -> p[1], pts), minimum(p -> p[2], pts), minimum(p -> p[3], pts)) .- R
    hi = Point3f(maximum(p -> p[1], pts), maximum(p -> p[2], pts), maximum(p -> p[3], pts)) .+ R
    dims = Vec{3,Int32}(max.(ceil.(Int32, (hi .- lo) ./ R), Int32(1)))
    ncells = Int(dims[1]) * Int(dims[2]) * Int(dims[3])
    cellof(p) = metaball_cell(dims, cellcoords(lo, R, dims, p))
    counts = zeros(Int32, ncells + 1)
    for p in pts
        counts[cellof(p) + 1] += Int32(1)
    end
    # counts[c + 1] is cell c's count, so the running sum to c is the points before c.
    cellstart = cumsum(counts) .+ Int32(1)
    next = copy(cellstart)
    sorted = similar(pts)
    for p in pts
        c = cellof(p)
        sorted[next[c]] = p
        next[c] += Int32(1)
    end
    dpoints = Mantle.devicearray(backend, Point3f, length(sorted)); copyto!(dpoints, sorted)
    dstart = Mantle.devicearray(backend, Int32, length(cellstart)); copyto!(dstart, cellstart)
    class = Mantle.devicearray(backend, UInt8, ncells)
    balls = Metaballs(dpoints, dstart, class, lo, dims, R, Float32(iso), Float32(step), nothing)
    mb = Metaballs(dpoints, dstart, class, lo, dims, R, Float32(iso), Float32(step), kernelshapes(anisotropy, balls, backend))
    # A launch per `chunk` cells, each its own submission. Where particles fill only a
    # band under the surface (a FLIP bake), nearly every occupied cell is undecided and
    # samples the field over its neighbours; one launch over all of them ran for seconds,
    # starved the desktop the GPU also drives, and the driver reset the device.
    for first in 0:chunk:ncells-1
        classify_kernel!(backend)(class, mb, Int32(first); ndrange = min(chunk, ncells - first))
    end
    return mb
end

Adapt.@adapt_structure Metaballs

"""
    translate(mb::Metaballs, v) -> Metaballs

The same metaballs moved by `v`: their points on the device and their grid with them,
so nothing is sorted or classified again.
"""
translate(mb::Metaballs, v::Vec3f) =
    Metaballs(mb.points .+ Ref(v), mb.cellstart, mb.class, mb.lo + v, mb.dims, mb.radius, mb.iso, mb.step, mb.shapes)

"""
    Anisotropic(; ratio = 4, neighbours = 25)

Kernels shaped by the particles round them (Yu & Turk 2013, "Reconstructing surfaces of
particle-based fluids using anisotropic kernels"): each particle's kernel is an
ellipsoid along the principal axes of its neighbours' weighted spread, its longest axis
the metaballs' radius (so it reaches no further than a ball, and the cells still hold
all that touches them) and the others shorter by the ratio of the spread's standard
deviations along them, down to `1/ratio` of it. (Yu & Turk's ratio of the variances,
unsmoothed positions given, thinned kernels until the surface tore into dark holes at a
whirlpool's lip.) A particle in a flat layer of water gets a flat kernel and the
surface over the layer is flat, where balls left it lumpy at their own spacing; thin
sheets and filaments stay thin. A particle with fewer than `neighbours` within the
radius keeps a ball. Each kernel is scaled by the ratio of a ball's volume to its own,
as Yu & Turk's `det G`, so that it adds as much to the field as a ball would.
"""
Base.@kwdef struct Anisotropic
    ratio::Float32 = 4f0
    neighbours::Int32 = 25
end

"""
    KernelShapes(shapes, scales)

Every particle's kernel (see [`Anisotropic`](@ref)): the symmetric matrix `M` whose
`dᵀ M d` takes the place of `|d|²/R²` in Wyvill's polynomial, and the kernel's scale.
"""
struct KernelShapes{M<:AbstractVector{Mat3f},S<:AbstractVector{Float32}}
    shapes::M
    scales::S
end
Adapt.@adapt_structure KernelShapes

kernelshapes(::Nothing, mb, backend) = nothing
function kernelshapes(a::Anisotropic, mb::Metaballs, backend)
    n = length(mb.points)
    shapes = Mantle.devicearray(backend, Mat3f, n)
    scales = Mantle.devicearray(backend, Float32, n)
    n > 0 && kernelshape_kernel!(backend)(shapes, scales, mb, a.ratio, a.neighbours; ndrange = n)
    return KernelShapes(shapes, scales)
end

"""
The eigenvalues (largest first) and unit eigenvectors (the columns) of the symmetric
`C`, by cyclic Jacobi rotations.
"""
@inline function symeigen(C::Mat3f)
    a11 = C[1, 1]; a22 = C[2, 2]; a33 = C[3, 3]; a12 = C[1, 2]; a13 = C[1, 3]; a23 = C[2, 3]
    v11 = 1f0; v12 = 0f0; v13 = 0f0; v21 = 0f0; v22 = 1f0; v23 = 0f0; v31 = 0f0; v32 = 0f0; v33 = 1f0
    for _ in 1:8
        # Rotate away a12, then a13, then a23 (each a plane rotation of rows and columns).
        if abs(a12) > 1f-20
            θ = (a22 - a11) / (2f0 * a12); t = sign(θ) / (abs(θ) + sqrt(θ * θ + 1f0)); t = θ == 0f0 ? 1f0 : t
            c = 1f0 / sqrt(t * t + 1f0); s = t * c
            a11, a22 = a11 - t * a12, a22 + t * a12; a12 = 0f0
            a13, a23 = c * a13 - s * a23, s * a13 + c * a23
            v11, v12 = c * v11 - s * v12, s * v11 + c * v12
            v21, v22 = c * v21 - s * v22, s * v21 + c * v22
            v31, v32 = c * v31 - s * v32, s * v31 + c * v32
        end
        if abs(a13) > 1f-20
            θ = (a33 - a11) / (2f0 * a13); t = sign(θ) / (abs(θ) + sqrt(θ * θ + 1f0)); t = θ == 0f0 ? 1f0 : t
            c = 1f0 / sqrt(t * t + 1f0); s = t * c
            a11, a33 = a11 - t * a13, a33 + t * a13; a13 = 0f0
            a12, a23 = c * a12 - s * a23, s * a12 + c * a23
            v11, v13 = c * v11 - s * v13, s * v11 + c * v13
            v21, v23 = c * v21 - s * v23, s * v21 + c * v23
            v31, v33 = c * v31 - s * v33, s * v31 + c * v33
        end
        if abs(a23) > 1f-20
            θ = (a33 - a22) / (2f0 * a23); t = sign(θ) / (abs(θ) + sqrt(θ * θ + 1f0)); t = θ == 0f0 ? 1f0 : t
            c = 1f0 / sqrt(t * t + 1f0); s = t * c
            a22, a33 = a22 - t * a23, a33 + t * a23; a23 = 0f0
            a12, a13 = c * a12 - s * a13, s * a12 + c * a13
            v12, v13 = c * v12 - s * v13, s * v12 + c * v13
            v22, v23 = c * v22 - s * v23, s * v22 + c * v23
            v32, v33 = c * v32 - s * v33, s * v32 + c * v33
        end
    end
    λ = Vec3f(a11, a22, a33)
    V = Mat3f(v11, v21, v31, v12, v22, v32, v13, v23, v33)
    # Largest first.
    i = λ[1] >= λ[2] ? (λ[1] >= λ[3] ? 1 : 3) : (λ[2] >= λ[3] ? 2 : 3)
    k = λ[1] < λ[2] ? (λ[1] < λ[3] ? 1 : 3) : (λ[2] < λ[3] ? 2 : 3)
    j = 6 - i - k
    return Vec3f(λ[i], λ[j], λ[k]), Mat3f(V[1, i], V[2, i], V[3, i], V[1, j], V[2, j], V[3, j], V[1, k], V[2, k], V[3, k])
end

"""
Particle `i`'s kernel (see [`Anisotropic`](@ref)): the spread of its neighbours within
the radius, weighted by `1 - (d/R)³`, about their weighted mean, and from its principal
axes the ellipsoid.
"""
@kernel function kernelshape_kernel!(shapes, scales, mb, ratio::Float32, needed::Int32)
    i = @index(Global)
    p = @inbounds mb.points[i]
    R = mb.radius
    c = cellcoords(mb.lo, R, mb.dims, p)
    sw = 0f0; m = Vec3f(0); xx = 0f0; xy = 0f0; xz = 0f0; yy = 0f0; yz = 0f0; zz = 0f0; n = Int32(0)
    for dk in Int32(-1):Int32(1), dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
        nb = c + Vec{3,Int32}(di, dj, dk)
        (Int32(0) <= nb[1] < mb.dims[1] && Int32(0) <= nb[2] < mb.dims[2] && Int32(0) <= nb[3] < mb.dims[3]) || continue
        cell = metaball_cell(mb.dims, nb)
        for q in @inbounds(mb.cellstart[cell]):(@inbounds(mb.cellstart[cell + 1]) - Int32(1))
            δ = Vec3f(@inbounds(mb.points[q]) - p)
            d = sqrt(sum(abs2, δ))
            if d < R
                w = 1f0 - (d / R)^3
                sw += w; m += w * δ; n += Int32(1)
                xx += w * δ[1] * δ[1]; xy += w * δ[1] * δ[2]; xz += w * δ[1] * δ[3]
                yy += w * δ[2] * δ[2]; yz += w * δ[2] * δ[3]; zz += w * δ[3] * δ[3]
            end
        end
    end
    ball = Mat3f(1f0 / (R * R), 0f0, 0f0, 0f0, 1f0 / (R * R), 0f0, 0f0, 0f0, 1f0 / (R * R))
    if n < needed
        @inbounds shapes[i] = ball
        @inbounds scales[i] = 1f0
    else
        m = m / sw
        C = Mat3f(xx / sw - m[1] * m[1], xy / sw - m[1] * m[2], xz / sw - m[1] * m[3],
                  xy / sw - m[1] * m[2], yy / sw - m[2] * m[2], yz / sw - m[2] * m[3],
                  xz / sw - m[1] * m[3], yz / sw - m[2] * m[3], zz / sw - m[3] * m[3])
        σ, V = symeigen(C)
        if σ[1] > 0f0
            a2 = R * max(sqrt(max(σ[2], 0f0) / σ[1]), 1f0 / ratio)
            a3 = R * max(sqrt(max(σ[3], 0f0) / σ[1]), 1f0 / ratio)
            D = Mat3f(1f0 / (R * R), 0f0, 0f0, 0f0, 1f0 / (a2 * a2), 0f0, 0f0, 0f0, 1f0 / (a3 * a3))
            @inbounds shapes[i] = V * D * V'
            @inbounds scales[i] = R * R / (a2 * a3)
        else
            @inbounds shapes[i] = ball
            @inbounds scales[i] = 1f0
        end
    end
end

# A particle's share of the field at offset `δ` from it, its gradient there, and the
# bounds on its share over a cell given the cell's nearest and farthest points' squared
# distances: by its ball, or by its ellipsoid (whose longest axis is the radius, so
# `dᵀ M d` is at least `|d|²/R²` and at most `trace(M) |d|²`).
@inline kernelvalue(::Nothing, mb, q, δ) = blob(sum(abs2, δ), mb.radius)
@inline function kernelvalue(k::KernelShapes, mb, q, δ)
    u = 1f0 - dot(δ, @inbounds(k.shapes[q]) * δ)
    return u > 0f0 ? @inbounds(k.scales[q]) * u * u * u : 0f0
end
@inline function kernelgradient(::Nothing, mb, q, δ)
    R2 = mb.radius * mb.radius
    u = 1f0 - sum(abs2, δ) / R2
    return u > 0f0 ? (-6f0 * u * u / R2) * δ : Vec3f(0)
end
@inline function kernelgradient(k::KernelShapes, mb, q, δ)
    Mδ = @inbounds(k.shapes[q]) * δ
    u = 1f0 - dot(δ, Mδ)
    return u > 0f0 ? (-6f0 * @inbounds(k.scales[q]) * u * u) * Mδ : Vec3f(0)
end
@inline kernelmost(::Nothing, mb, q, near2) = blob(near2, mb.radius)
@inline kernelmost(k::KernelShapes, mb, q, near2) = @inbounds(k.scales[q]) * blob(near2, mb.radius)
@inline kernelleast(::Nothing, mb, q, far2) = blob(far2, mb.radius)
@inline function kernelleast(k::KernelShapes, mb, q, far2)
    M = @inbounds k.shapes[q]
    u = 1f0 - (M[1, 1] + M[2, 2] + M[3, 3]) * far2
    return u > 0f0 ? @inbounds(k.scales[q]) * u * u * u : 0f0
end

"The cell coordinates (from 0) of `p`, clamped to the grid."
@inline function cellcoords(lo, R, dims, p)
    i = clamp(unsafe_trunc(Int32, floor((p[1] - lo[1]) / R)), Int32(0), dims[1] - Int32(1))
    j = clamp(unsafe_trunc(Int32, floor((p[2] - lo[2]) / R)), Int32(0), dims[2] - Int32(1))
    k = clamp(unsafe_trunc(Int32, floor((p[3] - lo[3]) / R)), Int32(0), dims[3] - Int32(1))
    return Vec{3,Int32}(i, j, k)
end

"The 1-based index of the cell at coordinates `c`."
@inline metaball_cell(dims, c) = Int(c[1]) + Int(dims[1]) * (Int(c[2]) + Int(dims[2]) * Int(c[3])) + 1

"A particle's contribution at squared distance `d2` (radius `R`): `(1 - d²/R²)³`, zero beyond."
@inline function blob(d2, R)
    q = 1f0 - d2 / (R * R)
    return q > 0f0 ? q * q * q : 0f0
end

"""
    field(mb, x) -> Float32

The metaball field at `x`.
"""
@inline function field(mb::Metaballs, x)
    c = cellcoords(mb.lo, mb.radius, mb.dims, x)
    s = 0f0
    for dk in Int32(-1):Int32(1), dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
        n = c + Vec{3,Int32}(di, dj, dk)
        (Int32(0) <= n[1] < mb.dims[1] && Int32(0) <= n[2] < mb.dims[2] && Int32(0) <= n[3] < mb.dims[3]) || continue
        cell = metaball_cell(mb.dims, n)
        for q in @inbounds(mb.cellstart[cell]):(@inbounds(mb.cellstart[cell + 1]) - Int32(1))
            s += kernelvalue(mb.shapes, mb, q, Vec3f(x - @inbounds(mb.points[q])))
        end
    end
    return s
end

"""
    gradient(mb, x) -> Vec3f

The gradient of the metaball field at `x`; the surface's outward normal is its opposite.
"""
@inline function gradient(mb::Metaballs, x)
    c = cellcoords(mb.lo, mb.radius, mb.dims, x)
    g = Vec3f(0)
    for dk in Int32(-1):Int32(1), dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
        n = c + Vec{3,Int32}(di, dj, dk)
        (Int32(0) <= n[1] < mb.dims[1] && Int32(0) <= n[2] < mb.dims[2] && Int32(0) <= n[3] < mb.dims[3]) || continue
        cell = metaball_cell(mb.dims, n)
        for q in @inbounds(mb.cellstart[cell]):(@inbounds(mb.cellstart[cell + 1]) - Int32(1))
            g += kernelgradient(mb.shapes, mb, q, Vec3f(x - @inbounds(mb.points[q])))
        end
    end
    return g
end

"""
Class every cell by bounds on the field within it: the least each particle can add
is its contribution at the cell's corner farthest from it, the most at the cell's
point nearest it. Above `iso` everywhere is inside, below it everywhere outside; of
the rest, a cell whose field is above `1.5 iso` all over a 3×3×3 lattice is inside
too, and only what is left can hold the surface.

Most of a body of water is deep inside it, and there the bounds decide nothing (the
far corners are out of most particles' reach) and the lattice costs 27 sums of
hundreds of particles. A cell whose 27 neighbours each hold as many particles as a
cell of fluid whose field is `1.5 iso` (`n` particles a cell make a field of
`n WYVILL_VOLUME` where they are spread as a fluid's are) is classed inside by the
counts alone, the lattice's test asked of the particles' numbers.
"""
@kernel function classify_kernel!(class, mb, first::Int32)
    idx = @index(Global)
    c0 = Int32(idx - 1) + first
    nx, ny = mb.dims[1], mb.dims[2]
    c = Vec{3,Int32}(c0 % nx, (c0 ÷ nx) % ny, c0 ÷ (nx * ny))
    lo = mb.lo + Vec3f(c) * mb.radius
    hi = lo + Vec3f(mb.radius)
    crowd = 1.5f0 * mb.iso / WYVILL_VOLUME
    crowded = true
    for dk in Int32(-1):Int32(1), dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
        n = c + Vec{3,Int32}(di, dj, dk)
        if Int32(0) <= n[1] < mb.dims[1] && Int32(0) <= n[2] < mb.dims[2] && Int32(0) <= n[3] < mb.dims[3]
            cell = metaball_cell(mb.dims, n)
            crowded &= Float32(@inbounds(mb.cellstart[cell + 1]) - @inbounds(mb.cellstart[cell])) >= crowd
        else
            crowded = false
        end
    end
    least = 0f0; most = 0f0
    for dk in Int32(-1):Int32(1), dj in Int32(-1):Int32(1), di in Int32(-1):Int32(1)
        crowded && break
        n = c + Vec{3,Int32}(di, dj, dk)
        (Int32(0) <= n[1] < mb.dims[1] && Int32(0) <= n[2] < mb.dims[2] && Int32(0) <= n[3] < mb.dims[3]) || continue
        cell = metaball_cell(mb.dims, n)
        for q in @inbounds(mb.cellstart[cell]):(@inbounds(mb.cellstart[cell + 1]) - Int32(1))
            p = @inbounds mb.points[q]
            near = clamp.(p, lo, hi)
            far = Point3f(ifelse(p[1] - lo[1] > hi[1] - p[1], lo[1], hi[1]),
                          ifelse(p[2] - lo[2] > hi[2] - p[2], lo[2], hi[2]),
                          ifelse(p[3] - lo[3] > hi[3] - p[3], lo[3], hi[3]))
            most += kernelmost(mb.shapes, mb, q, sum(abs2, p - near))
            least += kernelleast(mb.shapes, mb, q, sum(abs2, p - far))
        end
    end
    cls = crowded || least > mb.iso ? METABALL_INSIDE : most < mb.iso ? METABALL_OUTSIDE : METABALL_SURFACE
    if cls == METABALL_SURFACE
        # The bounds leave a cell deep in the water undecided: its far corners are out
        # of reach of most particles near it. Sampled on a 3×3×3 lattice, a cell whose
        # field is well above `iso` everywhere is water through and through; what that
        # can miss is a bubble smaller than half a cell.
        deep = true
        for k in 0:2, j in 0:2, i in 0:2
            x = lo + Vec3f(i, j, k) * (0.5f0 * mb.radius)
            if field(mb, x) < 1.5f0 * mb.iso
                deep = false
            end
        end
        deep && (cls = METABALL_INSIDE)
    end
    @inbounds class[c0 + Int32(1)] = cls
end

"""
    metaball_hit(mb, o, d, tmin, tmax) -> (hit, t, normal)

The first crossing of the metaball surface along the ray `o + t d` (`d` need not be
unit length) in `(tmin, tmax)`, from inside out or outside in, whichever side `o + tmin d`
is on, and the surface's outward unit normal there.
"""
@inline function metaball_hit(mb::Metaballs, o, d, tmin::Float32, tmax::Float32)
    t0, t1 = clipgrid(mb, o, d, tmin, tmax)
    t0 < t1 || return (false, tmax, Vec3f(0, 0, 1))
    inside = field(mb, o + tmin * d) > mb.iso
    R = mb.radius
    # The cell the ray enters at, and for each axis the `t` of the next cell boundary
    # (Amanatides & Woo).
    c = cellcoords(mb.lo, R, mb.dims, o + t0 * d)
    ci = c[1]; cj = c[2]; ck = c[3]
    si = d[1] >= 0f0 ? Int32(1) : Int32(-1); sj = d[2] >= 0f0 ? Int32(1) : Int32(-1); sk = d[3] >= 0f0 ? Int32(1) : Int32(-1)
    nx = boundary_t(mb.lo[1], R, ci, si, o[1], d[1]); ny = boundary_t(mb.lo[2], R, cj, sj, o[2], d[2])
    nz = boundary_t(mb.lo[3], R, ck, sk, o[3], d[3])
    ex = abs(d[1]) < 1f-12 ? Inf32 : R / abs(d[1]); ey = abs(d[2]) < 1f-12 ? Inf32 : R / abs(d[2])
    ez = abs(d[3]) < 1f-12 ? Inf32 : R / abs(d[3])
    dt = mb.step / sqrt(sum(abs2, d))
    ta = t0
    tprev = t0
    while ta < t1
        tb = min(min(nx, ny), min(nz, t1))
        class = @inbounds mb.class[metaball_cell(mb.dims, Vec{3,Int32}(ci, cj, ck))]
        if class == METABALL_SURFACE
            t = max(ta, tprev)
            while t < tb
                tn = min(t + dt, tb)
                if (field(mb, o + tn * d) > mb.iso) != inside
                    return crossing(mb, o, d, t, tn, inside)
                end
                t = tn
            end
            tprev = tb
        elseif (class == METABALL_INSIDE) != inside
            # A sure cell on the other side: the crossing lies between the last
            # sample and the cell's entry.
            return crossing(mb, o, d, tprev, ta, inside)
        else
            tprev = tb
        end
        # On to the next cell, across whichever boundary comes first.
        if nx <= ny && nx <= nz
            ci += si; nx += ex
            Int32(0) <= ci < mb.dims[1] || break
        elseif ny <= nz
            cj += sj; ny += ey
            Int32(0) <= cj < mb.dims[2] || break
        else
            ck += sk; nz += ez
            Int32(0) <= ck < mb.dims[3] || break
        end
        ta = tb
    end
    return (false, tmax, Vec3f(0, 0, 1))
end

"The part `(t0, t1)` of `(tmin, tmax)` where the ray is inside the grid; empty if `t0 >= t1`."
@inline function clipgrid(mb::Metaballs, o, d, tmin::Float32, tmax::Float32)
    hi = mb.lo + Vec3f(mb.dims) * mb.radius
    t0 = tmin; t1 = tmax
    for a in 1:3
        if abs(d[a]) < 1f-12
            (o[a] < mb.lo[a] || o[a] > hi[a]) && return (1f0, 0f0)
        else
            ta = (mb.lo[a] - o[a]) / d[a]; tb = (hi[a] - o[a]) / d[a]
            t0 = max(t0, min(ta, tb)); t1 = min(t1, max(ta, tb))
        end
    end
    return (t0, t1)
end

"The `t` at which the ray `o + t d` along one axis leaves cell `c` stepping `s`."
@inline function boundary_t(lo, R, c, s, o, d)
    abs(d) < 1f-12 && return Inf32
    return (lo + (Float32(c) + (s > Int32(0) ? 1f0 : 0f0)) * R - o) / d
end

"""
The crossing between `t0` (on the side `inside` says) and `t1` (on the other), found by
bisection, and the outward normal there.
"""
@inline function crossing(mb::Metaballs, o, d, t0::Float32, t1::Float32, inside::Bool; bisections = 10)
    for _ in 1:bisections
        mid = 0.5f0 * (t0 + t1)
        if (field(mb, o + mid * d) > mb.iso) != inside
            t1 = mid
        else
            t0 = mid
        end
    end
    g = gradient(mb, o + t1 * d)
    gl = sqrt(sum(abs2, g))
    return (true, t1, gl > 0f0 ? -g / gl : -Vec3f(d) / sqrt(sum(abs2, d)))
end

# ── In the tracer ───────────────────────────────────────────────────────────
#
# The same route `fem.jl` takes (see there and `Mantle/src/raytracing/accel.jl`): one
# AABB per cell that can hold the surface goes into a hardware acceleration structure,
# the traversal offers each box a ray passes, `procedural_candidate` finds the surface
# in that box's stretch of the ray, and the winning hit comes back as the exact tangent
# plane there, so nothing downstream knows the water was not triangles.

import Mantle: procedural_miss, procedural_candidate, procedural_commit, procedural_bary,
               candidate_primitive_index, candidate_object_ray, commit_intersection!

"""
    Everywhere()

Every crossing of the metaball field is surface.
"""
struct Everywhere end

"""
    Cylinder(centre, radius, floor)

Crossings are surface within `radius` of `centre` (in x and y) and above `floor`.
"""
struct Cylinder
    centre::Point2f
    radius::Float32
    floor::Float32
end

"""
    counts(region, p) -> Bool

Whether a crossing of the metaball field at `p` (object space) is surface. Beyond it the
particles' water runs on into water the scene draws otherwise, a sea around them or the
deep below them, with no surface between. Add a method for a region of your own.
"""
@inline counts(::Everywhere, p) = true
@inline function counts(c::Cylinder, p)
    dx = p[1] - c.centre[1]; dy = p[2] - c.centre[2]
    return dx * dx + dy * dy <= c.radius * c.radius && p[3] >= c.floor
end

"""
    MetaballSurface(interface, points, radius; iso, step = radius / 6, lattice = 4, region = Everywhere(), tile = 1)
    MetaballSurface(interface, mb::Metaballs; lattice = 4, region = Everywhere(), tile = 1)

A metaball surface as a material that carries its geometry (see [`Metaballs`](@ref)):
the surface at level `iso` of `points`' field, drawn with `interface` (a
`MediumInterface`, or any material), where `region` says crossings are surface (see
[`counts`](@ref)). Its textures (a bump, say) are laid over it from above, once every
`tile` metres in x and y. Given metaballs already built (for other uses of the same
field), it draws them as they are.

Rays find it in the field sampled `lattice` times across each cell that can hold the
surface and interpolated trilinearly, one look-up for the hundreds of particles the
field sums; the normal is the exact field's.
"""
struct MetaballSurface{P<:Union{AbstractVector{Point3f},Metaballs},R} <: GeneratedGeometry
    interface::Material
    points::P            # the particles, or their metaballs built already
    radius::Float32
    iso::Float32
    step::Float32
    lattice::Int32
    region::R
    tile::Float32
end

function MetaballSurface(interface::Material, points::AbstractVector{<:Point3}, radius::Real; iso::Real,
                         step::Real = radius / 6, lattice::Integer = 4, region = Everywhere(), tile::Real = 1)
    return MetaballSurface(interface, Point3f.(points), Float32(radius), Float32(iso), Float32(step), Int32(lattice),
                           region, Float32(tile))
end

MetaballSurface(interface::Material, mb::Metaballs; lattice::Integer = 4, region = Everywhere(), tile::Real = 1) =
    MetaballSurface(interface, mb, mb.radius, mb.iso, mb.step, Int32(lattice), region, Float32(tile))

shadingmaterial(m::MetaballSurface) = m.interface

"""
    tessellate(m::MetaballSurface) -> GeometryBasics.Mesh

A raster renderer's stand-in for the traced surface: in every cell that holds it and
whose region counts it, a square a cell and a half wide (so that squares on a slope
overlap rather than leave slits) where Newton's steps along the field's
gradient from the cell's centre meet the surface, facing the way the surface does. Not a
mesh of the iso-surface, which the tracer never needs: enough for a preview to show where
the water is and which way it faces.
"""
function tessellate(m::MetaballSurface)
    mb = m.points isa Metaballs ? m.points : Metaballs(m.points, m.radius; iso = m.iso, step = m.step)
    backend = KA.get_backend(mb.class)
    class = Array(mb.class)
    cells = Int32[c - 1 for c in eachindex(class) if class[c] == METABALL_SURFACE]
    pts = Mantle.devicearray(backend, Point3f, length(cells)); nrm = Mantle.devicearray(backend, Vec3f, length(cells))
    isempty(cells) || surfacesplat_kernel!(backend)(pts, nrm, Mantle.devicearray(backend, cells), mb; ndrange = length(cells))
    P = Array(pts); N = Array(nrm)
    h = 0.75f0 * mb.radius
    pos = Point3f[]; normals = Vec3f[]; faces = GeometryBasics.GLTriangleFace[]
    for (p, n) in zip(P, N)
        counts(m.region, p) || continue
        u = normalize(cross(abs(n[3]) < 0.9f0 ? Vec3f(0, 0, 1) : Vec3f(1, 0, 0), n)); v = cross(n, u)
        k = length(pos)
        append!(pos, (p - h * u - h * v, p + h * u - h * v, p + h * u + h * v, p - h * u + h * v))
        append!(normals, (n, n, n, n))
        append!(faces, (GeometryBasics.GLTriangleFace(k + 1, k + 2, k + 3), GeometryBasics.GLTriangleFace(k + 1, k + 3, k + 4)))
    end
    return GeometryBasics.Mesh(pos, faces; normal = normals)
end

"Surface cell `i`'s point on the surface, from its centre by Newton's steps along the gradient, and the outward normal there."
@kernel function surfacesplat_kernel!(pts, nrm, @Const(cells), mb)
    i = @index(Global)
    c = @inbounds cells[i]
    nx, ny = mb.dims[1], mb.dims[2]
    lo = mb.lo + Vec3f(c % nx, (c ÷ nx) % ny, c ÷ (nx * ny)) * mb.radius
    x = lo + Vec3f(0.5f0 * mb.radius)
    for _ in 1:4
        g = gradient(mb, x)
        gg = sum(abs2, g)
        if gg > 0f0
            x = x - ((field(mb, x) - mb.iso) / gg) * g
            x = clamp.(x, lo .- mb.radius, lo .+ 2f0 * mb.radius)
        end
    end
    g = gradient(mb, x)
    gl = sqrt(sum(abs2, g))
    @inbounds pts[i] = Point3f(x)
    @inbounds nrm[i] = gl > 0f0 ? Vec3f(-g / gl) : Vec3f(0, 0, 1)
end

# A plot's colour is for the raster stand-in (`tessellate`) only: tinting the traced
# surface's dielectric would colour what the water lets through.
merge_color_with_material(color_tex, m::MetaballSurface) = m

"The surface's metaballs on `backend`: built from its particles, or the ones it was given."
metaballs(m::MetaballSurface{<:AbstractVector}, backend) = Metaballs(m.points, m.radius; iso = m.iso, step = m.step, backend)
metaballs(m::MetaballSurface{<:Metaballs}, backend) = m.points

"""
The metaballs as the traversal sees them: the field, the cell each box stands for, the
medium interface every hit carries, and where crossings count as surface.
"""
struct MetaballElements{B<:Metaballs,C<:AbstractVector{Int32},F<:AbstractVector{Float32},R}
    balls::B
    cells::C            # the (0-based, linear) cell of box i
    bricks::F           # box i's field on a (lattice + 1)³ lattice, x fastest
    lattice::Int32
    interface::UInt32
    region::R
    tile::Float32       # metres per texture repeat, in x and y
end

# The region is adapted too: one may hold device arrays (a level set, say).
Adapt.adapt_structure(to, e::MetaballElements) =
    MetaballElements(Adapt.adapt(to, e.balls), Adapt.adapt(to, e.cells), Adapt.adapt(to, e.bricks), e.lattice,
                     e.interface, Adapt.adapt(to, e.region), e.tile)

"The exact field at every lattice point of every box (see [`MetaballElements`](@ref))."
@kernel function fieldbricks_kernel!(bricks, @Const(cells), mb::Metaballs, lattice::Int32)
    i = @index(Global)
    m = lattice + Int32(1)
    per = m * m * m
    b = (Int32(i) - Int32(1)) ÷ per
    l = (Int32(i) - Int32(1)) % per
    c = @inbounds cells[b + Int32(1)]
    cell = Vec{3,Int32}(c % mb.dims[1], (c ÷ mb.dims[1]) % mb.dims[2], c ÷ (mb.dims[1] * mb.dims[2]))
    q = Vec{3,Int32}(l % m, (l ÷ m) % m, l ÷ (m * m))
    @inbounds bricks[i] = field(mb, Point3f(mb.lo + (Vec3f(cell) + Vec3f(q) / Float32(lattice)) * mb.radius))
end

"""
    field(e::MetaballElements, first, lo, x) -> Float32

The field at `x` in the box whose cell starts at `lo` and whose lattice starts at
`first` in `e.bricks`, trilinearly between the lattice's samples.
"""
@inline function field(e::MetaballElements, first::Int32, lo::Point3f, x)
    n = e.lattice
    s = clamp.(Vec3f(x - lo) / e.balls.radius * Float32(n), 0f0, Float32(n))
    i = min.(unsafe_trunc.(Int32, s), n - Int32(1))
    f = s - Vec3f(i)
    k = n + Int32(1)
    at(di, dj, dk) = @inbounds e.bricks[first + (i[1] + di) + k * ((i[2] + dj) + k * (i[3] + dk))]
    x00 = at(0, 0, 0) * (1f0 - f[1]) + at(1, 0, 0) * f[1]
    x10 = at(0, 1, 0) * (1f0 - f[1]) + at(1, 1, 0) * f[1]
    x01 = at(0, 0, 1) * (1f0 - f[1]) + at(1, 0, 1) * f[1]
    x11 = at(0, 1, 1) * (1f0 - f[1]) + at(1, 1, 1) * f[1]
    y0 = x00 * (1f0 - f[2]) + x10 * f[2]
    y1 = x01 * (1f0 - f[2]) + x11 * f[2]
    return y0 * (1f0 - f[3]) + y1 * f[3]
end

"The traversal's running best hit: its `t` and where it is, in object space."
struct MetaballHit
    t::Float32
    x::Float32
    y::Float32
    z::Float32
end

@inline procedural_miss(::MetaballElements) = MetaballHit(1f30, 0f0, 0f0, 0f0)

@inline function procedural_candidate(e::MetaballElements, best::MetaballHit)
    mb = e.balls
    o, d = candidate_object_ray()
    o = Point3f(o[1], o[2], o[3]); d = Vec3f(d[1], d[2], d[3])
    box = Int32(candidate_primitive_index())
    cell = Int32(@inbounds e.cells[box])
    k = e.lattice + Int32(1)
    first = (box - Int32(1)) * k * k * k + Int32(1)
    nx, ny = mb.dims[1], mb.dims[2]
    lo = mb.lo + Vec3f(cell % nx, (cell ÷ nx) % ny, cell ÷ (nx * ny)) * mb.radius
    hi = lo + Vec3f(mb.radius)
    len = sqrt(sum(abs2, d))
    # Not at the ray's very origin: a ray leaving the surface starts on it.
    ta = 2f-3 / len; tb = best.t
    for a in 1:3
        if abs(d[a]) < 1f-12
            (o[a] < lo[a] || o[a] > hi[a]) && return best
        else
            t1 = (lo[a] - o[a]) / d[a]; t2 = (hi[a] - o[a]) / d[a]
            ta = max(ta, min(t1, t2)); tb = min(tb, max(t1, t2))
        end
    end
    ta < tb || return best
    dt = mb.step / len
    inside = field(e, first, lo, o + ta * d) > mb.iso
    t = ta
    while t < tb
        tn = min(t + dt, tb)
        if (field(e, first, lo, o + tn * d) > mb.iso) != inside
            # Bisect to the crossing.
            l = t; h = tn
            for _ in 1:10
                m = 0.5f0 * (l + h)
                if (field(e, first, lo, o + m * d) > mb.iso) != inside
                    h = m
                else
                    l = m
                end
            end
            p = o + h * d
            if counts(e.region, p)
                commit_intersection!(h)
                return MetaballHit(h, p[1], p[2], p[3])
            end
            # Not surface here: carry on from the other side.
            inside = !inside
        end
        t = tn
    end
    return best
end

@inline function procedural_commit(e::MetaballElements, best::MetaballHit, prim_idx, t, empty)
    best.t >= 1f29 && return empty
    p = Point3f(best.x, best.y, best.z)
    g = gradient(e.balls, p)
    gl = sqrt(sum(abs2, g))
    n = gl > 0f0 ? Vec3f(-g / gl) : Vec3f(0, 0, 1)
    # A tangent frame on the plane through the hit, `uv` where the hit is in x and y (in
    # tiles): `du` the tangent with no y in it, so on a surface facing up (or down) moving
    # along `du` moves along u, and a bump's slopes face the way its texture says.
    a = abs(n[2]) < 0.9f0 ? Vec3f(0, 1, 0) : Vec3f(1, 0, 0)
    du = normalize(cross(a, n)); dv = cross(n, du)
    uv0 = Point2f(p[1], p[2]) / e.tile
    md = TriangleMeta(e.interface, UInt32(1), UInt32(0))
    return typeof(empty)(
        SVector{3, Point3f}(p, p + e.tile * du, p + e.tile * dv),
        SVector{3, Normal3f}(n, n, n),
        SVector{3, Vec3f}(du, du, du),
        SVector{3, Point2f}(uv0, uv0 + Point2f(1, 0), uv0 + Point2f(0, 1)),
        md)
end

@inline procedural_bary(::MetaballElements, ::MetaballHit) = SVector{3, Float32}(1f0, 0f0, 0f0)

Base.push!(scene::Scene, ::GeometryBasics.Mesh, m::MetaballSurface; kw...) = push!(scene, m; kw...)
Base.push!(scene::Scene, ::Any, m::MetaballSurface; kw...) = push!(scene, m; kw...)

function Base.push!(scene::Scene, m::MetaballSurface; transform = Mat4f(LinearAlgebra.I))
    interface = m.interface isa MediumInterface ? m.interface : MediumInterface(m.interface)
    return push!(scene, m, push!(scene, interface); transform)
end

"""
    push!(scene, mesh, mi::UInt32, m::MetaballSurface; transform, area_lights) -> SceneHandle

The surface's geometry again, drawn with the medium interface already stored in slot
`mi` (which [`update_material!`](@ref) has brought up to date): RayMakie's rebuild of a
plot whose metaballs changed. Pushing a fresh interface every frame left the last one's
air, its bricks and its majorants (a gigabyte for a whirlpool) in the scene's sets.
"""
Base.push!(scene::Scene, ::GeometryBasics.Mesh, mi::UInt32, m::MetaballSurface;
           transform = Mat4f(LinearAlgebra.I), area_lights = SetKey[]) = push!(scene, m, mi; transform)

"""
    rebuilds_with_material(material) -> Bool

Whether a new `material` in a plot is new geometry (a `GeneratedGeometry` that carries
its own), which a renderer rebuilds into the plot's material slot with
`push!(scene, mesh, slot, material)`, rather than a new look for the same geometry.
"""
rebuilds_with_material(::Material) = false
rebuilds_with_material(::MetaballSurface) = true

function Base.push!(scene::Scene, m::MetaballSurface, mi::UInt32; transform = Mat4f(LinearAlgebra.I))
    accel = scene.accel
    Mantle.supports_procedural_traversal(accel) || error("""
        this scene cannot trace procedural geometry, so a `MetaballSurface` would render as \
        nothing: `Mantle.supports_procedural_traversal` says `false` for $(nameof(typeof(accel))). \
        Build the scene with `hw_accel = true` on a device with ray queries.""")
    balls = metaballs(m, accel.backend)
    class = Array(balls.class)
    cells = Int32[c - 1 for c in eachindex(class) if class[c] == METABALL_SURFACE]
    nx, ny = Int(balls.dims[1]), Int(balls.dims[2])
    aabbs = map(cells) do c
        lo = balls.lo + Vec3f(c % nx, (c ÷ nx) % ny, c ÷ (nx * ny)) * balls.radius
        Mantle.AABB(Point3f(lo), Point3f(lo .+ balls.radius))
    end
    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device())) do c
        Mantle.build_blas_aabb(c, aabbs)
    end
    handle = push!(accel, blas, transform; instance_id = UInt32(0))
    dcells = Mantle.devicearray(accel.backend, cells)
    k = Int(m.lattice) + 1
    bricks = Mantle.devicearray(accel.backend, Float32, length(cells) * k^3)
    isempty(cells) || fieldbricks_kernel!(accel.backend)(bricks, dcells, balls, m.lattice; ndrange = length(bricks))
    accel.procedural = MetaballElements(balls, dcells, bricks, m.lattice, UInt32(mi), m.region, m.tile)
    return SceneHandle(scene, mi, handle)
end

# The stored material of a metaball surface is its interface (its geometry is rebuilt
# by `push!` into the slot). Two methods, not one on a `Union`: the slot-index method
# for any material is equally specific on the key and Julia calls that a tie.
update_material!(scene::Scene, key::UInt32, m::MetaballSurface) = update_material!(scene, key, m.interface)
update_material!(scene::Scene, key::MediumInterfaceIdx, m::MetaballSurface) = update_material!(scene, key, m.interface)

# ─── water with air in it ────────────────────────────────────────────────────

"""
    AeratedMedium(foam, bubbles; bounds, radius, σ_a, σ_s, onset, full, foamσ, air, g = 0.8, airg = 0.8,
                  merge = radius / 3, lattice = 3, ior = 1.33, weight = Returns(1), backend)

The medium inside a [`MetaballSurface`](@ref) of water, with the whitewater's air in it
as particles, the way the water's surface is its particles: the water absorbs `σ_a`
and scatters `σ_s` (per metre, RGB) everywhere in `bounds`, and the air scatters white
on top. Each particle stands for its share of air over `radius` (Wyvill's kernel, as
the surface's, normalised to particles per cubic metre). The `bubbles` are a haze:
`air` square metres of cross-section each. The `foam`, which floats in a thin layer on
the water, is all bubbles or none: where its density rises from `onset` to `full`
particles per cubic metre it turns to foam scattering `foamσ` per metre, the iso level
of its particles as the water's surface is of its own. `g` is the Henyey-Greenstein
asymmetry of the water's sediment. A particle at `p` stands for `weight(p)` of one: a
weight falling to zero towards the edge of the water drawn fades the air out there
instead of cutting it off.

The air scatters forward too, as `airg` says, and dense foam scatters hundreds of times
before light gets out of it; it is drawn by the similarity relation instead (Zhao,
Ramamoorthi & Bala 2014): scattering `σ_s (1 - airg)` with no preferred direction, which
diffuses light the same in a fifth of the scattering events. The trackers take the
water as the control of decomposition tracking (see `control`) and look up only the
air.

The air's densities are the particles' kernel sums, cached: particles closer than
`merge` are merged into one at their centre, the cells `radius` wide over `bounds` that
any particle reaches get a brick of the sums at `lattice + 1` points a side, evaluated
on `backend`, and a lookup interpolates its brick trilinearly. A sum over the
particles took hundreds of terms where the foam crowds, at every step a ray took
through it; the brick is one fetch, and its largest sample bounds the cell exactly,
which makes the cell's majorant.

The water lies under a surface of index `ior`: light sampling inside it samples the
sky through Snell's window (see `snells_window`).
"""
struct AeratedMedium{I,S,M<:MajorantGrid} <: Medium
    brick::I             # per cell: its brick's first sample, 0 where no air reaches
    samples::S           # (foam, bubbles) per cubic metre, (lattice + 1)³ per brick, x fastest
    lo::Point3f
    dims::Vec{3,Int32}
    radius::Float32      # the cells' size, and the kernel's
    lattice::Int32       # brick intervals a side
    σ_a::RGBSpectrum
    σ_s::RGBSpectrum
    onset::Float32
    full::Float32
    foamσ::Float32       # the foam's scattering, reduced by the similarity relation
    air::Float32         # a bubble's cross-section, reduced likewise
    g::Float32
    base::Float32        # the water's share of every cell's majorant (see `controlmajorant`)
    ior::Float32         # its surface's index of refraction (see `refracting`)
    bounds::Bounds3
    majorant_grid::M
end


"The foam's share of the water at foam density `ρ`: none below `onset`, all from `full`."
@inline function foamlayer(ρ, onset, full)
    s = clamp((ρ - onset) / (full - onset), 0f0, 1f0)
    return s * s * (3f0 - 2f0 * s)
end

"""
    mergeair(foam, bubbles, lo, hi, size; weight = Returns(1)) -> (points, weights)

The particles in the box `lo`–`hi` merged into bins `size` wide: one point per bin at
its particles' centre, weighted by how many foam and bubble particles it stands for,
each particle at `p` as `weight(p)` of one.
"""
function mergeair(foam, bubbles, lo::Point3f, hi::Point3f, size::Float32; weight = Returns(1f0))
    n = Vec{3,Int}(max.(ceil.(Int, (hi .- lo) ./ size), 1))
    bin(p) = (q = clamp.(floor.(Int, (p .- lo) ./ size), 0, n .- 1); q[1] + n[1] * (q[2] + n[2] * q[3]))
    inside(p) = all(lo .<= p .< hi)
    items = [(Point3f(p), Vec2f(weight(Point3f(p)), 0)) for p in foam if inside(Point3f(p))]
    append!(items, [(Point3f(p), Vec2f(0, weight(Point3f(p)))) for p in bubbles if inside(Point3f(p))])
    keys = [bin(p) for (p, _) in items]
    order = sortperm(keys)
    points = Point3f[]; weights = Vec2f[]
    i = 1
    while i <= length(order)
        j = i; centre = Vec3f(0); w = Vec2f(0)
        while j <= length(order) && keys[order[j]] == keys[order[i]]
            p, k = items[order[j]]
            centre += Vec3f(p); w += k
            j += 1
        end
        push!(points, Point3f(centre / (j - i))); push!(weights, w)
        i = j
    end
    return points, weights
end

"""
    AirParticles

The merged air particles sorted into the medium's cells, as the kernel that fills the
bricks reads them.
"""
struct AirParticles{P,W,I}
    points::P
    weights::W
    cellstart::I
    lo::Point3f
    dims::Vec{3,Int32}
    radius::Float32
    perblob::Float32     # particles per cubic metre a kernel's peak stands for
end
Adapt.@adapt_structure AirParticles

"The foam's and the bubbles' particles per cubic metre at `x`: a sum over the cells within a kernel's reach."
@inline function airdensity(a::AirParticles, x)
    c0 = cellcoords(a.lo, a.radius, a.dims, x .- a.radius)
    c1 = cellcoords(a.lo, a.radius, a.dims, x .+ a.radius)
    f = 0f0; u = 0f0
    for k in c0[3]:c1[3], j in c0[2]:c1[2], i in c0[1]:c1[1]
        cell = metaball_cell(a.dims, Vec{3,Int32}(i, j, k))
        for q in @inbounds(a.cellstart[cell]):(@inbounds(a.cellstart[cell + 1]) - Int32(1))
            v = blob(sum(abs2, x - @inbounds(a.points[q])), a.radius)
            if v > 0f0
                w = @inbounds a.weights[q]
                f += v * w[1]; u += v * w[2]
            end
        end
    end
    return Vec2f(f * a.perblob, u * a.perblob)
end

"Brick sample `i` (from 1): the air's densities at its lattice point in its cell."
@kernel function airbricks_kernel!(samples, @Const(cells), a::AirParticles, lattice::Int32)
    i = @index(Global)
    m = lattice + Int32(1)
    per = m * m * m
    b = (Int32(i) - Int32(1)) ÷ per
    l = (Int32(i) - Int32(1)) % per
    c = @inbounds cells[b + Int32(1)]          # the cell, from 0
    cell = Vec{3,Int32}(c % a.dims[1], (c ÷ a.dims[1]) % a.dims[2], c ÷ (a.dims[1] * a.dims[2]))
    q = Vec{3,Int32}(l % m, (l ÷ m) % m, l ÷ (m * m))
    x = a.lo + (Vec3f(cell) + Vec3f(q) / Float32(lattice)) * a.radius
    @inbounds samples[i] = airdensity(a, Point3f(x))
end

function AeratedMedium(foam::AbstractVector{<:Point3}, bubbles::AbstractVector{<:Point3}; bounds::Bounds3, radius::Real,
                       σ_a, σ_s, onset::Real, full::Real, foamσ::Real, air::Real, g::Real = 0.8f0, airg::Real = 0.8f0,
                       merge::Real = radius / 3, lattice::Integer = 3, ior::Real = 1.33f0, weight = Returns(1f0), backend)
    R = Float32(radius)
    reduced = 1f0 - Float32(airg)
    foamσ = Float32(foamσ) * reduced
    air = Float32(air) * reduced
    lo = Point3f(bounds.p_min)
    dims = Vec{3,Int32}(max.(ceil.(Int32, (Point3f(bounds.p_max) .- lo) ./ R), Int32(1)))
    hi = lo .+ Vec3f(dims) .* R
    pts, wts = mergeair(foam, bubbles, lo, hi, Float32(merge); weight)
    ncells = Int(dims[1]) * Int(dims[2]) * Int(dims[3])
    cellof(p) = metaball_cell(dims, cellcoords(lo, R, dims, p))
    counts = zeros(Int32, ncells + 1)
    for p in pts
        counts[cellof(p) + 1] += Int32(1)
    end
    cellstart = cumsum(counts) .+ Int32(1)
    next = copy(cellstart)
    sorted = similar(pts); sortedw = similar(wts)
    for (p, w) in zip(pts, wts)
        c = cellof(p)
        sorted[next[c]] = p; sortedw[next[c]] = w
        next[c] += Int32(1)
    end
    # The cells any particle reaches get a brick.
    reached = falses(ncells)
    nx, ny, nz = Int(dims[1]), Int(dims[2]), Int(dims[3])
    for p in sorted
        c = cellcoords(lo, R, dims, p)
        for dk in -1:1, dj in -1:1, di in -1:1
            i = Int(c[1]) + di; j = Int(c[2]) + dj; k = Int(c[3]) + dk
            (0 <= i < nx && 0 <= j < ny && 0 <= k < nz) || continue
            boxlo = lo .+ Vec3f(i, j, k) .* R
            near = clamp.(p, boxlo, boxlo .+ R)
            sum(abs2, p - near) < R * R && (reached[i + nx * (j + ny * k) + 1] = true)
        end
    end
    cells = Int32[c - 1 for c in 1:ncells if reached[c]]
    m = Int(lattice) + 1
    per = m^3
    brick = zeros(Int32, ncells)
    for (b, c) in enumerate(cells)
        brick[c + 1] = Int32((b - 1) * per + 1)
    end
    perblob = 1f0 / (WYVILL_VOLUME * R^3)
    particles = AirParticles(Mantle.devicearray(backend, sorted), Mantle.devicearray(backend, sortedw),
                             Mantle.devicearray(backend, cellstart), lo, dims, R, perblob)
    dsamples = Mantle.devicearray(backend, Vec2f, length(cells) * per)
    isempty(cells) || airbricks_kernel!(backend)(dsamples, Mantle.devicearray(backend, cells), particles, Int32(lattice);
                                                 ndrange = length(cells) * per)
    samples = Array(dsamples)
    water = RGBSpectrum(σ_a...) + RGBSpectrum(σ_s...)
    # A tenth over the water's largest RGB extinction: its spectrum, uplifted, may
    # rise above that a little at some wavelengths.
    base = 1.1f0 * max(water.c[1], water.c[2], water.c[3])
    # A cell's majorant: the water, and its brick's densest samples, which bound
    # whatever trilinear interpolation between them gives.
    grid = MajorantGrid(dims, Vector{Float32})
    fill!(grid.voxels, base)
    for (b, c) in enumerate(cells)
        top = reduce((x, y) -> max.(x, y), view(samples, (b - 1) * per + 1:b * per))
        grid.voxels[c + 1] = base + foamσ * foamlayer(top[1], Float32(onset), Float32(full)) + air * top[2]
    end
    return AeratedMedium(brick, samples, lo, dims, R, Int32(lattice), RGBSpectrum(σ_a...), RGBSpectrum(σ_s...),
                         Float32(onset), Float32(full), foamσ, air, Float32(g), base, Float32(ior), Bounds3(lo, hi), grid)
end

Adapt.@adapt_structure AeratedMedium

@inline get_template_grid(medium::AeratedMedium) = medium.majorant_grid
@propagate_inbounds is_emissive(::AeratedMedium) = false

"The air's densities at `x`: its cell's brick, interpolated trilinearly; none where no air reaches."
@inline function airdensity(m::AeratedMedium, brick, samples, x)
    c = cellcoords(m.lo, m.radius, m.dims, x)
    first = @inbounds brick[metaball_cell(m.dims, c)]
    first == Int32(0) && return Vec2f(0)
    n = m.lattice
    s = clamp.((Vec3f(x - m.lo) / m.radius - Vec3f(c)) * Float32(n), 0f0, Float32(n))
    i = min.(unsafe_trunc.(Int32, s), n - Int32(1))
    f = s - Vec3f(i)
    k = n + Int32(1)
    at(di, dj, dk) = @inbounds samples[first + (i[1] + di) + k * ((i[2] + dj) + k * (i[3] + dk))]
    x00 = at(0, 0, 0) * (1f0 - f[1]) + at(1, 0, 0) * f[1]
    x10 = at(0, 1, 0) * (1f0 - f[1]) + at(1, 1, 0) * f[1]
    x01 = at(0, 0, 1) * (1f0 - f[1]) + at(1, 0, 1) * f[1]
    x11 = at(0, 1, 1) * (1f0 - f[1]) + at(1, 1, 1) * f[1]
    y0 = x00 * (1f0 - f[2]) + x10 * f[2]
    y1 = x01 * (1f0 - f[2]) + x11 * f[2]
    return y0 * (1f0 - f[3]) + y1 * f[3]
end

"The air's white scattering at `x`, per metre."
@inline function airscattering(medium::AeratedMedium, media, x)
    d = airdensity(medium, Raycore.deref(media, medium.brick), Raycore.deref(media, medium.samples), x)
    return medium.foamσ * foamlayer(d[1], medium.onset, medium.full) + medium.air * d[2]
end

@propagate_inbounds function sample_point(medium::AeratedMedium, media, table::RGBToSpectrumTable, p::Point3f,
                                          λ::Wavelengths)::MediumProperties
    white = airscattering(medium, media, p)
    σ_a = uplift_rgb_unbounded(table, medium.σ_a, λ)
    σ_s = uplift_rgb_unbounded(table, medium.σ_s + RGBSpectrum(white), λ)
    # The air's scattering has no preferred direction (see above); the water's does.
    water = max(medium.σ_s.c[1], medium.σ_s.c[2], medium.σ_s.c[3])
    g = water * medium.g / (water + white + 1f-12)
    return MediumProperties(σ_a, σ_s, SpectralRadiance(0f0), g)
end

# Decomposition tracking (see `control`): the water is the control, the air the residual.
@inline control(medium::AeratedMedium, media, table::RGBToSpectrumTable, λ::Wavelengths) =
    MediumProperties(uplift_rgb_unbounded(table, medium.σ_a, λ), uplift_rgb_unbounded(table, medium.σ_s, λ),
                     SpectralRadiance(0f0), medium.g)
@inline controlmajorant(medium::AeratedMedium) = medium.base
# Water under its metaball surface: light reaches it through Snell's window.
@inline refracting(medium::AeratedMedium) = medium.ior

"The air at `p`: white, scattering without a preferred direction (the similarity relation)."
@propagate_inbounds residual(medium::AeratedMedium, media, table::RGBToSpectrumTable, p::Point3f,
                             λ::Wavelengths)::MediumProperties =
    MediumProperties(SpectralRadiance(0f0), SpectralRadiance(airscattering(medium, media, p)), SpectralRadiance(0f0), 0f0)

@propagate_inbounds function get_majorant(medium::AeratedMedium, table::RGBToSpectrumTable, ray::Raycore.Ray,
                                          t_min::Float32, t_max::Float32, λ::Wavelengths)::RayMajorantSegment
    top = 0f0
    for v in medium.majorant_grid.voxels
        top = max(top, v)
    end
    return RayMajorantSegment(t_min, t_max, SpectralRadiance(top))
end

@propagate_inbounds function create_majorant_iterator(medium::AeratedMedium, table::RGBToSpectrumTable,
                                                      ray::Raycore.Ray, t_max::Float32, λ::Wavelengths)
    t_enter, t_exit = ray_bounds_intersect(ray.o, ray.d, medium.bounds)
    t_enter = max(t_enter, 0f0)
    t_exit = min(t_exit, t_max)
    t_enter >= t_exit && return RayMajorantIterator(medium.majorant_grid)
    # The majorant grid holds the whole σ_t bound per cell.
    dda = create_dda_iterator(medium.majorant_grid, medium.bounds, ray.o, ray.d, t_enter, t_exit, SpectralRadiance(1f0))
    return RayMajorantIterator(dda)
end

@propagate_inbounds create_majorant_iterator(medium::AeratedMedium, table::RGBToSpectrumTable, ray::Raycore.Ray,
                                             t_max::Float32, λ::Wavelengths, ::MajorantGrid) =
    create_majorant_iterator(medium, table, ray, t_max, λ)
