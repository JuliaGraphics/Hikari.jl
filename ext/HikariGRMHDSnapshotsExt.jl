module HikariGRMHDSnapshotsExt
import Hikari
import GRMHDSnapshots
using GRMHDSnapshots: map_plasma, bunit, rhounit, volume_ratio
using StaticArrays: SVector
using Hikari: EmissionParams, electron_density, c₅, K, synchrotron_αν,
    emissivity_grid, emission_contribution_grid

# x^E for a quarter-integer E (the p slider snaps to ½ steps → emission exponents are quarters), else the
# exact `x^E`. E is a compile-time `Val`, so this folds to a few `sqrt`s — no libm `pow` in the hot loop.
@inline function _powq(x, ::Val{E}) where {E}
    i = floor(Int, E); f = E - i
    f == 0    ? x^i :
    f == 0.25 ? x^i * sqrt(sqrt(x)) :
    f == 0.5  ? x^i * sqrt(x) :
    f == 0.75 ? x^i * sqrt(x * sqrt(x)) :
                x^E                                       # non-quarter exponent → exact fallback
end

# Per-cell synchrotron emissivity jν over the snapshot grid (:φ,:θ,:r), σ-cut applied — fused into one
# threaded pass over `map_plasma` (physics + electron density + jν + mask, no intermediate arrays). The
# cell-independent ν^(-(p-1)/2) is folded into the prefactor. The magnetization cut keeps emission where
# b² ≤ σcut·ρ (Synchray's form): a Bool factor that strong-zeros the funnel and any ρ≤0 cells.
function Hikari.emissivity_grid(snap::GRMHDSnapshots.GRMHDSnapshot, ep::EmissionParams)
    (; ne, p, ν, γmin, γmax, σcut) = ep
    Cj = c₅(p)*K(p, γmin, γmax)*ν^(-(p - 1)/2)
    _emissivity_grid(snap, ne, Cj, bunit(snap), rhounit(snap), p, γmin, σcut, Val((p + 1)/2))
end

function Hikari.absorption_grid(snap::GRMHDSnapshots.GRMHDSnapshot, ep::EmissionParams)
    (; ne, p, ν, γmin, γmax, σcut) = ep
    Bu, ρu = bunit(snap), rhounit(snap)
    # Evaluate only the cell-independent coefficient on the host.
    Cα = synchrotron_αν(1.0, 1.0, p, ν, γmin, γmax)
    map_plasma(snap, snap.rho) do ps, _, (ρ,)
        Bg = sqrt(max(ps.bsq, 0.0)) * Bu
        ps.bsq <= σcut * ρ && ρ > 0 ?
            Cα * electron_density(ne, Bg, ρ, ρu, p, γmin) * Bg^((p+2)/2) : 0.0
    end
end
@inline function _emissivity_grid(snap, ne, Cj, Bunit, ρunit, p, γmin, σcut, ::Val{EX}) where {EX}
    map_plasma(snap, snap.rho) do ps, _, (ρ,)
        Bg = sqrt(ps.bsq)*Bunit
        Cj*electron_density(ne, Bg, ρ, ρunit, p, γmin)*_powq(Bg, Val(EX)) * (ps.bsq <= σcut*ρ)
    end
end

# Emissivity × gravitational-redshift boost g^((p+5)/2), g=√(−g_tt) (Liouville g³ × frequency shift
# g^((p-1)/2)), fused into the same single pass. `zg` is the per-cell `grav_redshift` from `map_plasma`;
# it is 0 inside the ergosphere (no static emitter), which zeroes that region's contribution.
function Hikari.emission_contribution_grid(snap::GRMHDSnapshots.GRMHDSnapshot, ep::EmissionParams)
    (; ne, p, ν, γmin, γmax, σcut) = ep
    Cj = c₅(p)*K(p, γmin, γmax)*ν^(-(p - 1)/2)
    _emission_contribution_grid(snap, ne, Cj, bunit(snap), rhounit(snap), p, γmin, σcut, Val((p + 1)/2), Val((p + 5)/2))
end
@inline function _emission_contribution_grid(snap, ne, Cj, Bunit, ρunit, p, γmin, σcut, ::Val{EX}, ::Val{GEX}) where {EX,GEX}
    map_plasma(snap, snap.rho) do ps, zg, (ρ,)
        Bg = sqrt(ps.bsq)*Bunit
        Cj*electron_density(ne, Bg, ρ, ρunit, p, γmin)*_powq(Bg, Val(EX)) * (ps.bsq <= σcut*ρ) * _powq(zg, Val(GEX))
    end
end

# Contribution to the observed image per *displayed* (Cartesian) volume: emission_contribution × the
# proper-to-coordinate volume ratio √γ/(r²sinθ). Integrating this over the drawn dx dy dz box recovers
# the observed flux (optically thin, no ray bending, gravitational redshift only) — unlike
# emission_contribution_grid, which is a density per *proper* volume.
Hikari.image_contribution_grid(snap::GRMHDSnapshots.GRMHDSnapshot, ep::EmissionParams) = emission_contribution_grid(snap, ep) .* volume_ratio(snap)

# The fluid's Cartesian velocity in the local static Schwarzschild frame, from
# the relative-velocity primitive `u` at (rg, θ, φ), rg in r_g (a = 0), capped
# at 0.999c. Relative velocity -> orthonormal Eulerian frame, then a radial
# Lorentz boost into the static frame. On and inside the horizon there is no
# static frame: everything falls in at c, the limit the boost reaches at rg = 2.
function static_velocity(u, rg, theta, phi)
    st, ct = sincos(theta); sp, cp = sincos(phi)
    rg <= 2 && return -0.999f0 * SVector{3,Float32}(st*cp, st*sp, ct)
    ur, ut, up = u[1]*sqrt(1+2/rg), rg*u[2], rg*st*u[3]
    gamma = sqrt(1+ur^2+ut^2+up^2)
    vr, vt, vp = ur/gamma, ut/gamma, up/gamma
    bn = -2/rg
    den = 1+bn*vr
    vr = (vr+bn)/den
    vt *= sqrt(1-bn^2)/den
    vp *= sqrt(1-bn^2)/den
    v = SVector{3,Float32}(vr*st*cp + vt*ct*cp - vp*sp,
              vr*st*sp + vt*ct*sp + vp*cp, vr*ct - vt*st)
    return v * min(1f0, 0.999f0/max(sqrt(sum(abs2,v)), 1f-8))
end

function load_radiative_snapshot(path, mass_unit, black_hole_mass)
    isfinite(mass_unit) && mass_unit>0 || throw(ArgumentError("mass_unit must be positive CGS grams"))
    isfinite(black_hole_mass) && black_hole_mass>0 || throw(ArgumentError("black_hole_mass must be positive CGS grams"))
    isfile(path) || error("Missing GRMHD snapshot: $path; fetch the declared sequence before rendering")
    snap = GRMHDSnapshots.load_snapshot(path; M_unit = mass_unit, MBH = black_hole_mass)
    iszero(snap.spin) || error("This medium's transport requires a spin-zero snapshot")
    return snap
end

# Exposure is a display choice, not a fitted flux measurement: emissivity keeps
# its relative physical values, scaled by `normalization` or its own peak.
function normalize_emissivity!(j, normalization)
    peak = maximum(j)
    peak > 0 || error("GRMHD snapshot has no emitting plasma in the scene")
    reference = isnothing(normalization) ? peak : normalization
    isfinite(reference) && reference > 0 || error("Invalid emissivity normalization")
    j ./= reference
    return peak
end

function Hikari.native_radiative_volume(path::AbstractString; mass_unit, black_hole_mass,
                                        fade = (11f0, 14f0), normalization = nothing,
                                        emission_params = Hikari.EmissionParams())
    inner, outer = Float32.(fade)
    0 < inner < outer || throw(ArgumentError("fade must be two increasing positive radii"))
    snap = load_radiative_snapshot(path, mass_unit, black_hole_mass)
    radii = snap.r_prof ./ 2
    steps = diff(log.(radii))
    all(s -> isapprox(s, steps[1]; rtol = 1e-6), steps) ||
        error("Native sampling needs a log-uniform radial grid")
    nr = findfirst(>=(outer), radii)
    nr === nothing && error("The snapshot grid ends inside the fade")
    native_j = Hikari.emissivity_grid(snap, emission_params)
    native_a = Hikari.absorption_grid(snap, emission_params)
    nphi, ntheta = size(snap.rho, 1), size(snap.rho, 2)
    j = zeros(Float32, nphi, ntheta, nr)
    alpha = similar(j)
    velocity = similar(j, SVector{3,Float32})
    rs_cm = 2GRMHDSnapshots.lunit(snap)
    Threads.@threads for i in 1:nr
        s = clamp((radii[i]-inner)/(outer-inner), 0, 1)
        taper = 1 - s^2*(3 - 2s)
        for jt in 1:ntheta, k in 1:nphi
            j[k,jt,i] = Float32(max(native_j[k,jt,i], 0) * taper)
            alpha[k,jt,i] = Float32(max(native_a[k,jt,i], 0) * taper * rs_cm)
            velocity[k,jt,i] = static_velocity(snap.velrel[k,jt,i], 2radii[i], snap.th_grid[i,jt],
                                               snap.φ0 + (k-1)*snap.dφ)
        end
    end
    peak = normalize_emissivity!(j, normalization)
    return (; density=j, absorption=alpha, velocity, theta=Float32.(snap.th_grid[1:nr, :]),
            log_r0=Float32(log(radii[1])), dlog_r=Float32(steps[1]),
            phi0=Float32(snap.φ0), dphi=Float32(snap.dφ),
            path=abspath(path), normalization, emissivity_peak=peak, extent=Float32(radii[nr]))
end

# Cartesian snapshot preparation shared by file-backed editor inputs and scenes.
# Coordinates and extent are in Schwarzschild radii (2 r_g).
function Hikari.radiative_volume(path::AbstractString; mass_unit, black_hole_mass,
                                 resolution = 192, extent = 15f0, normalization=nothing,
                                 emission_params=Hikari.EmissionParams())
    resolution>=2 || throw(ArgumentError("volume resolution must be at least two"))
    isfinite(extent) && extent>1.025 || throw(ArgumentError("invalid volume extent"))
    G = GRMHDSnapshots
    snap = load_radiative_snapshot(path, mass_unit, black_hole_mass)
    ep = emission_params
    native_j = Hikari.emissivity_grid(snap, ep)
    native_a = Hikari.absorption_grid(snap, ep)
    j = zeros(Float32, resolution, resolution, resolution)
    alpha = similar(j)
    velocity = fill(zero(SVector{3,Float32}), size(j))
    rs_cm = 2G.lunit(snap)
    Threads.@threads for k in 1:resolution
        for iy in 1:resolution, ix in 1:resolution
            p = SVector{3,Float32}(ix-1, iy-1, k-1) .* (2extent/(resolution-1)) .- extent
            r = sqrt(sum(abs2,p))
            if r <= 1.025f0 || r >= 14f0
                alpha[ix,iy,k] = 0f0
                continue
            end
            x, y, z = 2 .* Float64.(p)
            rg = 2Float64(r)
            theta = acos(clamp(z/rg, -1, 1))
            phi = atan(y,x)
            taper = 1 - clamp((r-11f0)/3f0, 0f0, 1f0)^2 * (3 - 2clamp((r-11f0)/3f0, 0f0, 1f0))
            # Keep the module binding constant inside the threaded closure.
            # Capturing local `G::Module` makes every voxel lookup dynamic.
            j[ix,iy,k] = Float32(max(GRMHDSnapshots.sample_field(snap, native_j, x,y,z), 0) * taper)
            alpha[ix,iy,k] = Float32(max(GRMHDSnapshots.sample_field(snap, native_a, x,y,z), 0) * taper * rs_cm)
            u = GRMHDSnapshots.sample_field(snap, snap.velrel, x,y,z)
            velocity[ix,iy,k] = static_velocity(u, rg, theta, phi)
        end
    end
    # Absorption retains the stated CGS units.
    peak = normalize_emissivity!(j, normalization)
    # Bound each trilinear cell jointly. Multiplying the GLOBAL densest cell
    # by the GLOBAL fastest cell is safe but grossly pessimistic: the fastest
    # infall is not where absorption peaks. Convex interpolation bounds both
    # alpha and |v| by the maxima of the same eight corners.
    slab_bounds=zeros(Float32,resolution-1)
    # EmissionParams.p is dynamically typed. Resolve this invariant once,
    # rather than boxing arithmetic and exponentiation in every grid cell.
    majorant_exponent = Float32((ep.p+2)/4)
    Threads.@threads for k in 1:resolution-1
        bound=0f0
        @inbounds for iy in 1:resolution-1, ix in 1:resolution-1
            amax=0f0; v2max=0f0
            for dz in 0:1, dy in 0:1, dx in 0:1
                v=velocity[ix+dx,iy+dy,k+dz]
                amax=max(amax,alpha[ix+dx,iy+dy,k+dz])
                v2max=max(v2max,sum(abs2,v))
            end
            vmax=sqrt(v2max)
            bound=max(bound,amax*((1+vmax)/(1-vmax))^majorant_exponent)
        end
        slab_bounds[k]=bound
    end
    majorant = maximum(slab_bounds) * 1.0001f0 + 1f-6
    result = (; density=j, absorption=alpha, velocity, path=abspath(path), normalization,
              max_absorption=Float32(majorant), emissivity_peak=peak,
              extent=Float32(extent))
    return result
end

end
