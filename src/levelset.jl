# A surface traced straight from a NanoVDB grid (see nanotree.jl): where its values
# cross `iso`. Water reconstructed from a liquid simulation's particles, the iso surface
# of a smoke's density, anything a grid holds. Rays meet it the way NanoVDB's level-set
# tracer does (`zeroCrossing`, Museth 2021), with the hardware's traversal in place of
# the hierarchical DDA: one box per leaf the surface passes through goes into the
# acceleration structure, and in a box the ray samples the grid trilinearly half a
# voxel at a time, a sign change refined by bisection; the normal is the gradient of
# the interpolated grid by central differences, smooth across voxels. Its cost does not
# grow with how many particles the grid was made from.

"""
    LevelSetSurface(interface, tree::NanoTree; iso = 0, rising = false, region = Everywhere(), tile = 1)

The surface where `tree`'s values cross `iso`, drawn with `interface` (a
`MediumInterface`, or any material). `rising` says the values rise into what the surface
encloses (an indicator, a density); otherwise they fall into it (a signed distance,
negative inside). Crossings are surface where `region` says (see [`counts`](@ref)).
Textures lie over it from above once every `tile` metres. Where a leaf has no neighbour
the grid is taken to go on as at its edge: the band a grid was made in ends without a
surface there.
"""
struct LevelSetSurface{T<:NanoTree,R} <: GeneratedGeometry
    interface::Material
    tree::T
    iso::Float32
    rising::Bool
    region::R
    tile::Float32
end
LevelSetSurface(interface::Material, tree::NanoTree; iso::Real = 0, rising::Bool = false, region = Everywhere(), tile::Real = 1) =
    LevelSetSurface(interface, tree, Float32(iso), rising, region, Float32(tile))

shadingmaterial(m::LevelSetSurface) = m.interface
merge_color_with_material(color_tex, m::LevelSetSurface) = m
rebuilds_with_material(::LevelSetSurface) = true

"""
The value at voxel `ijk`, from leaf `leaf` (first word) when it lies in it, otherwise
from the tree; where no leaf holds it, the nearest voxel of `leaf` (the grid going on
as at the leaf's edge).
"""
@inline function bandvalue(t::NanoTree, words, leaf::Int32, origin::Vec{3,Int32}, ijk::Vec{3,Int32})
    local8 = ijk - origin
    if all(Int32(0) .<= local8 .< Int32(8))
        return wordfloat(words, leaf + NT_LEAF_VALUES + leaf_coord_to_offset((ijk[1], ijk[2], ijk[3])))
    end
    v, found = nanoleafvalue(t, words, ijk)
    found && return v
    c = clamp.(local8, Int32(0), Int32(7)) + origin
    return wordfloat(words, leaf + NT_LEAF_VALUES + leaf_coord_to_offset((c[1], c[2], c[3])))
end

"The grid at index-space point `q`, trilinear, as seen from leaf `leaf` (see `bandvalue`)."
@inline function bandsample(t::NanoTree, words, leaf::Int32, origin::Vec{3,Int32}, q::Vec3f)
    i = floor.(Int32, q)
    f = q - Vec3f(i)
    at(a, b, c) = bandvalue(t, words, leaf, origin, i + Vec{3,Int32}(a, b, c))
    x00 = at(0, 0, 0) * (1f0 - f[1]) + at(1, 0, 0) * f[1]
    x10 = at(0, 1, 0) * (1f0 - f[1]) + at(1, 1, 0) * f[1]
    x01 = at(0, 0, 1) * (1f0 - f[1]) + at(1, 0, 1) * f[1]
    x11 = at(0, 1, 1) * (1f0 - f[1]) + at(1, 1, 1) * f[1]
    return (x00 * (1f0 - f[2]) + x10 * f[2]) * (1f0 - f[3]) + (x01 * (1f0 - f[2]) + x11 * f[2]) * f[3]
end

"""
The level set as the traversal sees it: the grid, the leaf (first word) and index-space
corner of each box, the medium interface every hit carries, which way the values run.
"""
struct LevelSetElements{T<:NanoTree,L<:AbstractVector{Int32},O<:AbstractVector{Vec{3,Int32}},R}
    tree::T
    leaves::L           # box i's leaf, its first word
    origins::O          # box i's leaf's index-space corner
    iso::Float32
    rising::Bool
    interface::UInt32
    region::R
    tile::Float32
    opaque::Bool        # the interface stops shadow rays (see `opaque`)
end
Adapt.@adapt_structure LevelSetElements

"The traversal's running best hit: its `t` and where it is, in object space."
struct LevelSetHit
    t::Float32
    x::Float32
    y::Float32
    z::Float32
end

@inline procedural_miss(::LevelSetElements) = LevelSetHit(1f30, 0f0, 0f0, 0f0)
@inline procedural_opaque(e::LevelSetElements) = e.opaque

@inline function procedural_candidate(e::LevelSetElements, best::LevelSetHit, q)
    t = e.tree
    words = t.words
    ro, rd = candidate_object_ray(q)
    # New names, not `o, d` reassigned: the closure `f` below captures them, and a
    # captured variable assigned twice is boxed, which the device cannot type.
    o = Point3f(ro[1], ro[2], ro[3]); d = Vec3f(rd[1], rd[2], rd[3])
    box = Int32(candidate_primitive_index(q))
    leaf = @inbounds e.leaves[box]
    origin = @inbounds e.origins[box]
    lo = t.lo + Vec3f(origin) * t.voxel
    hi = lo + Vec3f(8f0 * t.voxel)
    len = sqrt(sum(abs2, d))
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
    f(s) = bandsample(t, words, leaf, origin, indexspace(t, o + s * d)) - e.iso
    dt = 0.5f0 * t.voxel / len
    above = f(ta) > 0f0
    s = ta
    while s < tb
        sn = min(s + dt, tb)
        if (f(sn) > 0f0) != above
            l = s; h = sn
            for _ in 1:12
                m = 0.5f0 * (l + h)
                if (f(m) > 0f0) != above
                    h = m
                else
                    l = m
                end
            end
            p = o + h * d
            counts(e.region, p) && return LevelSetHit(h, p[1], p[2], p[3])
            # Not surface here: carry on from the other side.
            above = !above
        end
        s = sn
    end
    return best
end

"The gradient of the interpolated grid at world point `p`, by central differences half a voxel either way (per metre)."
@inline function bandgradient(t::NanoTree, words, leaf::Int32, origin::Vec{3,Int32}, p)
    q = indexspace(t, p)
    g(a) = bandsample(t, words, leaf, origin, q + 0.5f0 * a) - bandsample(t, words, leaf, origin, q - 0.5f0 * a)
    return Vec3f(g(Vec3f(1, 0, 0)), g(Vec3f(0, 1, 0)), g(Vec3f(0, 0, 1))) / t.voxel
end

@inline function procedural_commit(e::LevelSetElements, best::LevelSetHit, prim_idx, s, empty)
    best.t >= 1f29 && return empty
    p = Point3f(best.x, best.y, best.z)
    box = Int32(prim_idx)
    gr = bandgradient(e.tree, e.tree.words, @inbounds(e.leaves[box]), @inbounds(e.origins[box]), p)
    gl = sqrt(sum(abs2, gr))
    # Out of what the surface encloses: against the values if they rise into it.
    n = gl > 0f0 ? Vec3f((e.rising ? -1f0 : 1f0) * gr / gl) : Vec3f(0, 0, 1)
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

@inline procedural_bary(::LevelSetElements, ::LevelSetHit) = SVector{3, Float32}(1f0, 0f0, 0f0)

"""
    crossedleaves(tree, iso) -> (leaves, origins)

The leaves (from 1, in the tree's order) the surface at `iso` can pass through: those
whose values reach both sides of it, and those on one side next to a leaf on the other
(the crossing then lies in the half voxel between their voxels); and every leaf's
index-space corner.
"""
function crossedleaves(t::NanoTree, iso::Float32)
    w = Array(t.words)
    n = Int(t.leaves)
    lo = Vector{Float32}(undef, n); hi = similar(lo); org = Vector{Vec{3,Int32}}(undef, n)
    for l in 1:n
        at = t.leaf + (l - 1) * NT_LEAF_WORDS
        org[l] = Vec{3,Int32}(reinterpret(Int32, w[at + 1]), reinterpret(Int32, w[at + 2]), reinterpret(Int32, w[at + 3]))
        lo[l] = reinterpret(Float32, w[at + NT_LEAF_MIN + 1]); hi[l] = reinterpret(Float32, w[at + NT_LEAF_MIN + 2])
    end
    index = Dict(o => l for (l, o) in enumerate(org))
    keep = Int32[]
    for l in 1:n
        crossed = lo[l] <= iso <= hi[l]
        if !crossed
            below = hi[l] < iso
            for dk in -1:1, dj in -1:1, di in -1:1
                m = get(index, org[l] + Vec{3,Int32}(8di, 8dj, 8dk), 0)
                m == 0 && continue
                if below ? hi[m] >= iso : lo[m] <= iso
                    crossed = true
                    break
                end
            end
        end
        crossed && push!(keep, Int32(l))
    end
    return keep, org
end

Base.push!(scene::Scene, ::GeometryBasics.Mesh, m::LevelSetSurface; kw...) = push!(scene, m; kw...)
Base.push!(scene::Scene, ::Any, m::LevelSetSurface; kw...) = push!(scene, m; kw...)

function Base.push!(scene::Scene, m::LevelSetSurface; transform = Mat4f(LinearAlgebra.I))
    interface = m.interface isa MediumInterface ? m.interface : MediumInterface(m.interface)
    return push!(scene, m, push!(scene, interface); transform)
end

Base.push!(scene::Scene, ::GeometryBasics.Mesh, mi::UInt32, m::LevelSetSurface;
           transform = Mat4f(LinearAlgebra.I), area_lights = SetKey[]) = push!(scene, m, mi; transform)

function Base.push!(scene::Scene, m::LevelSetSurface, mi::UInt32; transform = Mat4f(LinearAlgebra.I))
    accel = scene.accel
    Mantle.supports_procedural_traversal(accel) || error("""
        this scene cannot trace procedural geometry, so a `LevelSetSurface` would render as \
        nothing: `Mantle.supports_procedural_traversal` says `false` for $(nameof(typeof(accel))). \
        Build the scene with `hw_accel = true` on a device with ray queries.""")
    t = m.tree
    keep, org = crossedleaves(t, m.iso)
    aabbs = map(keep) do l
        lo = t.lo + Vec3f(org[l]) * t.voxel
        Mantle.AABB(Point3f(lo), Point3f(lo .+ 8f0 * t.voxel))
    end
    blas = Mantle.build_accel!(Mantle.batchqueue(Mantle.Device())) do c
        Mantle.build_blas_aabb(c, aabbs)
    end
    handle = push!(accel, blas, transform; instance_id = UInt32(0))
    leaves = Mantle.devicearray(accel.backend, Int32[t.leaf + (l - 1) * NT_LEAF_WORDS for l in keep])
    origins = Mantle.devicearray(accel.backend, org[keep])
    accel.procedural = LevelSetElements(t, leaves, origins, m.iso, m.rising, UInt32(mi), m.region, m.tile,
                                        opaque(m.interface))
    return SceneHandle(scene, mi, handle)
end

store_interface!(scene::Scene, mi::MediumInterfaceIdx, m::LevelSetSurface) = store_interface!(scene, mi, m.interface)

"""
A raster renderer's stand-in for the traced surface: a square where the grid crosses
`iso` between two neighbouring voxels of a leaf, facing along their axis. Enough for a
preview to show where the surface is.
"""
function tessellate(m::LevelSetSurface)
    t = m.tree
    w = Array(t.words)
    keep, org = crossedleaves(t, m.iso)
    pos = Point3f[]; normals = Vec3f[]; faces = GeometryBasics.GLTriangleFace[]
    h = 0.6f0 * t.voxel
    value(l, i, j, k) = reinterpret(Float32, w[t.leaf + (l - 1) * NT_LEAF_WORDS + NT_LEAF_VALUES + (i << 6 | j << 3 | k) + 1])
    for l in keep, i in 0:7, j in 0:7, k in 0:7, (a, step) in enumerate(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        i2, j2, k2 = i + step[1], j + step[2], k + step[3]
        (i2 < 8 && j2 < 8 && k2 < 8) || continue
        v1 = value(l, i, j, k) - m.iso; v2 = value(l, i2, j2, k2) - m.iso
        (v1 > 0) == (v2 > 0) && continue
        s = v1 / (v1 - v2)
        c = worldspace(t, Vec3f(org[l]) + Vec3f(i, j, k) + s * Vec3f(step...))
        n = Vec3f(step...) * ((v2 > v1) == m.rising ? -1f0 : 1f0)
        u = normalize(cross(abs(n[3]) < 0.9f0 ? Vec3f(0, 0, 1) : Vec3f(1, 0, 0), n)); v = cross(n, u)
        q = length(pos)
        append!(pos, (Point3f(c - h * u - h * v), Point3f(c + h * u - h * v), Point3f(c + h * u + h * v), Point3f(c - h * u + h * v)))
        append!(normals, (n, n, n, n))
        append!(faces, (GeometryBasics.GLTriangleFace(q + 1, q + 2, q + 3), GeometryBasics.GLTriangleFace(q + 1, q + 3, q + 4)))
    end
    return GeometryBasics.Mesh(pos, faces; normal = normals)
end
