# Natural water, from its measured optical properties, and the air in it.
#
# Seawater's colour is what its constituents absorb and scatter, wavelength by
# wavelength: pure water (Pope & Fry 1997, Segelstein 1981), its molecular scattering,
# which depends on temperature and salinity (Zhang, Hu & He 2009), phytoplankton
# (Bricaud et al. 1995; their scattering after Morel & Maritorena 2001), mineral
# particles a tidal current keeps in suspension and dissolved organic matter (Babin et
# al. 2003). Concentrations are the knobs, as an oceanographer would state them.
#
# Air in the water (bubbles), water films round air (foam) and water in the air
# (spray) are volume fractions on NanoVDB grids. By geometric optics a population of
# spheres of radius `r` taking up a fraction `φ` of a volume scatters `3φ/(2r)` per
# metre (extinction efficiency 2), scarcely absorbing: air does not absorb, and the
# water films of foam and a droplet's few millimetres of water barely do. Light leaves
# dense foam after hundreds of such events; the similarity relation (Zhao, Ramamoorthi
# & Bala 2014) draws it with the reduced coefficient `(1 - g) 3φ/(2r)`, scattering
# without a preferred direction.

include("water-data.jl")

"Linear interpolation in a measured table `(λ, a)`, clamped at its ends."
function tabulated(t, λ)
    λs = t.λ; i = searchsortedlast(λs, Float32(λ))
    i < 1 && return Float64(t.a[1])
    i >= length(λs) && return Float64(t.a[end])
    f = (λ - λs[i]) / (λs[i + 1] - λs[i])
    return (1 - f) * t.a[i] + f * t.a[i + 1]
end

"Pure water's absorption at `λ` nm (1/m): Pope & Fry's integrating cavity measurements where they reach, Segelstein's compilation outside."
purewater_absorption(λ) = 380 <= λ <= 727.5 ? tabulated(POPE_FRY_1997, λ) : tabulated(SEGELSTEIN_1981, λ)

"""
    seawater_scattering(λ; temperature = 15, salinity = 34, depolarization = 0.039) -> b (1/m)

Molecular scattering of pure seawater at `λ` nm, from the density and the concentration
fluctuations of water at `temperature` (°C) and `salinity` (PSU): Zhang, Hu & He 2009
(Opt. Express 17, 5698), a port of their `betasw_ZHH2009.m`. About 0.0029/m at 500 nm,
falling as λ^-4.3; half of it backscattered.
"""
function seawater_scattering(λ; temperature = 15.0, salinity = 34.0, depolarization = 0.039)
    T = Float64(temperature); S = Float64(salinity); δ = Float64(depolarization); λ = Float64(λ)
    Na = 6.0221417930e23; Kbz = 1.3806503e-23; Tk = T + 273.15; M0 = 18e-3
    # Refractive index of seawater (Quan & Fry 1994) in air (Ciddor 1996), and its salinity derivative.
    n_air = 1.0 + (5792105.0 / (238.0185 - 1 / (λ / 1e3)^2) + 167917.0 / (57.362 - 1 / (λ / 1e3)^2)) / 1e8
    n0 = 1.31405; n1 = 1.779e-4; n2 = -1.05e-6; n3 = 1.6e-8; n4 = -2.02e-6; n5 = 15.868; n6 = 0.01155
    n7 = -0.00423; n8 = -4382; n9 = 1.1455e6
    nsw = (n0 + (n1 + n2 * T + n3 * T^2) * S + n4 * T^2 + (n5 + n6 * S + n7 * T) / λ + n8 / λ^2 + n9 / λ^3) * n_air
    dnds = (n1 + n2 * T + n3 * T^2 + n6 / λ) * n_air
    # Isothermal compressibility from the secant bulk modulus (Millero 1980).
    kw = 19652.21 + 148.4206 * T - 2.327105 * T^2 + 1.360477e-2 * T^3 - 5.155288e-5 * T^4
    a0 = 54.6746 - 0.603459 * T + 1.09987e-2 * T^2 - 6.167e-5 * T^3
    b0 = 7.944e-2 + 1.6483e-2 * T - 5.3009e-4 * T^2
    isocomp = 1 / (kw + a0 * S + b0 * S^1.5) * 1e-5
    # Density of seawater (UNESCO 1981).
    ρw = 999.842594 + 6.793952e-2 * T - 9.09529e-3 * T^2 + 1.001685e-4 * T^3 - 1.120083e-6 * T^4 + 6.536332e-9 * T^5
    ρ = ρw + (8.24493e-1 - 4.0899e-3 * T + 7.6438e-5 * T^2 - 8.2467e-7 * T^3 + 5.3875e-9 * T^4) * S +
        (-5.72466e-3 + 1.0227e-4 * T - 1.6546e-6 * T^2) * S^1.5 + 4.8314e-4 * S^2
    # Salinity derivative of the log of water activity (Millero & Leung 1976).
    dlnawds = (-5.58651e-4 + 2.40452e-7 * T - 3.12165e-9 * T^2 + 2.40808e-11 * T^3) +
              1.5 * (1.79613e-5 - 9.9422e-8 * T + 2.08919e-9 * T^2 - 1.39872e-11 * T^3) * S^0.5 +
              2 * (-2.31065e-6 - 1.37674e-9 * T - 1.93316e-11 * T^2) * S
    # Density derivative of the refractive index (PMH model).
    n2w = nsw^2
    dfri = (n2w - 1) * (1 + 2 / 3 * (n2w + 2) * (nsw / 3 - 1 / 3 / nsw)^2)
    shape = (6 + 6δ) / (6 - 7δ)
    β_df = π^2 / 2 * (λ * 1e-9)^(-4) * Kbz * Tk * isocomp * dfri^2 * shape
    β_cf = 2π^2 * (λ * 1e-9)^(-4) * nsw^2 * (S * M0 * dnds^2 / ρ / (-dlnawds) / Na) * shape
    return 8π / 3 * (β_df + β_cf) * (2 + δ) / (1 + δ)
end

"""
    Seawater(; temperature = 15, salinity = 34, chlorophyll = 0.5, minerals = 1, cdom = 0.05)

Natural seawater: `temperature` (°C) and `salinity` (PSU) of the water, `chlorophyll`
(mg/m³) of its phytoplankton, `minerals` (g/m³) of suspended mineral particles (a tidal
race keeps sediment up), and `cdom`, the absorption (1/m) at 440 nm of its dissolved
organic matter. See [`absorption`](@ref), [`scattering`](@ref), [`backscattering`](@ref).
"""
Base.@kwdef struct Seawater
    temperature::Float64 = 15.0
    salinity::Float64 = 34.0
    chlorophyll::Float64 = 0.5
    minerals::Float64 = 1.0
    cdom::Float64 = 0.05
end

"Phytoplankton absorption at `λ` (1/m): Bricaud et al.'s `A(λ) Chl^(1-B)`, with their typical `B` of 0.3; none beyond 700 nm."
phytoplankton_absorption(w::Seawater, λ) = λ > 700 ? 0.0 : tabulated(BRICAUD_1995, λ) * w.chlorophyll^0.7

"The positive exponent for `(550/λ)^ν` in Case 1 water (Morel & Maritorena 2001, equations 12–13)."
morel_exponent(chl) = chl > 0 ? max(0.5 * (0.3 - log10(chl)), 0.0) : 0.0

"""
    absorption(w::Seawater, λ) -> 1/m

Pure water, phytoplankton, mineral particles (`0.041 m²/g` at 443 nm, slope 0.0123/nm)
and dissolved organic matter (slope 0.0176/nm): Babin et al. 2003 for the last two.
"""
absorption(w::Seawater, λ) = purewater_absorption(λ) + phytoplankton_absorption(w, λ) +
                             0.041 * w.minerals * exp(-0.0123 * (λ - 443)) + w.cdom * exp(-0.0176 * (λ - 440))

"""
    scattering(w::Seawater, λ) -> 1/m

Molecular scattering (see [`seawater_scattering`](@ref)), particles that come with
phytoplankton (`0.416 Chl^0.766`, Morel & Maritorena 2001) and mineral particles (`0.51
m²/g` at 555 nm, Babin et al. 2003, falling as λ^-0.5).
"""
scattering(w::Seawater, λ) = seawater_scattering(λ; w.temperature, w.salinity) +
                             0.416 * w.chlorophyll^0.766 * (550 / λ) +
                             0.51 * w.minerals * (555 / λ)^0.5

"""
    backscattering(w::Seawater, λ) -> 1/m

Half the molecular scattering, a fraction of the biogenic particles' (Morel & Maritorena
2001) and 1.9 % of the minerals' (Petzold's average particle phase function).
"""
function backscattering(w::Seawater, λ)
    chl = w.chlorophyll
    ν = morel_exponent(chl)
    bio = 0.416 * chl^0.766
    ratio = chl > 0 ? 0.002 + 0.01 * (0.5 - 0.25 * log10(chl)) * (550 / λ)^ν : 0.0
    return 0.5 * seawater_scattering(λ; w.temperature, w.salinity) + ratio * bio + 0.019 * 0.51 * w.minerals * (555 / λ)^0.5
end

"The fraction of light a Henyey-Greenstein phase function of asymmetry `g` scatters backwards."
hg_backfraction(g) = g < 1e-6 ? 0.5 : (1 - g) / (2g) * ((1 + g) / sqrt(1 + g^2) - 1)

"The Henyey-Greenstein asymmetry that scatters the fraction `B` backwards (bisection)."
function hg_asymmetry(B)
    lo, hi = 0.0, 0.9999
    for _ in 1:60
        m = (lo + hi) / 2
        hg_backfraction(m) > B ? (lo = m) : (hi = m)
    end
    return (lo + hi) / 2
end

"""
    WaterMedium(water; air = (), bounds, cell = 0.4, from = 360, to = 830, step = 5, ior = 1.34, backend)

A medium of `water` (a [`Seawater`](@ref), or `nothing` for air), its absorption and
scattering tabulated every `step` nm from `from` to `to` and sampled per wavelength,
scattering forward as the Henyey-Greenstein asymmetry with its backscattered fraction
at 550 nm. `air` holds what scatters white in it: `(tree, radius, g)` per kind, a
[`NanoTree`](@ref) of volume fraction (of air in water, of water in air) and the radius
and phase asymmetry of its bubbles or droplets (see the file's head). The water is the
control of decomposition tracking (see `control`), the air the residual, bounded per
`cell`-wide cell of `bounds` by their trilinear interpolation support.
`majorant_factor` subdivides these acceleration cells without changing the medium
bounds or physical fields. `ior` is the index of refraction of
the surface over it (see `refracting`).
"""
struct WaterMedium{V,A,M} <: Medium
    spectra::V           # σ_a at the tabulated wavelengths, then σ_s
    from::Float32
    step::Float32
    n::Int32
    g::Float32
    base::Float32        # the water's largest extinction (the control's majorant)
    air::A               # ((tree, σ per unit fraction), ...): white scattering on top
    ior::Float32
    bounds::Bounds3
    majorant_grid::M
end

"The volume fraction a [`NanoTree`](@ref) holds at `p`, trilinear; the tree's words looked up through `media`."
@inline fraction(media, t::NanoField, p) = nanofield_sample(t, media, p)

"The white scattering of `terms` at `p` (per metre, reduced by the similarity relation)."
@inline airscattering(media, ::Tuple{}, p) = 0f0
@inline airscattering(media, terms::Tuple, p) =
    first(terms)[2] * max(fraction(media, first(terms)[1], p), 0f0) + airscattering(media, Base.tail(terms), p)

function validate_water_majorants(layout)
    layout in (:dense,:tiled) && return nothing
    layout === :native && throw(ArgumentError("Native auxiliary majorants were retired: full-scene traversal is slower on NVIDIA and loses the AMD device. Use :dense; physical density fields remain NanoVDB."))
    throw(ArgumentError("majorants must be :dense or :tiled"))
end

function WaterMedium(water::Union{Seawater,Nothing}; air = (), bounds::Bounds3, cell = 0.4f0, from = 360, to = 830, step = 5,
                     ior = 1.34f0, majorants=:dense, majorant_factor::Integer=1, backend)
    validate_water_majorants(majorants)
    majorant_factor>=1 || throw(ArgumentError("majorant_factor must be positive"))
    λs = from:step:to
    σa = water === nothing ? zeros(Float32, length(λs)) : Float32[absorption(water, λ) for λ in λs]
    σs = water === nothing ? zeros(Float32, length(λs)) : Float32[scattering(water, λ) for λ in λs]
    g = water === nothing ? 0f0 : Float32(hg_asymmetry(backscattering(water, 550) / scattering(water, 550)))
    base = 1.01f0 * maximum(σa .+ σs)
    terms = Tuple((tree, Float32((1 - ga) * 1.5 / r)) for (tree, r, ga) in air)
    # The majorant per cell: the water, and the densest air whose leaves reach the cell
    # (a leaf's largest value bounds the trilinear lookup inside it and up to half a
    # voxel past it, so leaves are widened by a voxel).
    lo = Point3f(bounds.p_min); hi = Point3f(bounds.p_max)
    dims = Vec{3,Int32}(max.(ceil.(Int32, (hi .- lo) ./ Float32(cell)), Int32(1)))
    # Refine the acceleration cells after fixing the original physical bounds.
    # Neither the medium extent nor its sampled density/spectra changes.
    actualbounds = Bounds3(lo, lo .+ Vec3f(dims) .* Float32(cell))
    dims = dims .* Int32(majorant_factor)
    grid = MajorantGrid(dims, Vector{Float32})
    fill!(grid.voxels, base)
    for (tree, σ) in terms
        field = nanomajorant(tree, actualbounds, dims)
        grid.voxels .+= σ .* field.voxels
    end
    majorants === :tiled && (grid = tiledmajorants(grid))
    return WaterMedium(vcat(σa, σs), Float32(from), Float32(step), Int32(length(λs)), g, base, terms, Float32(ior),
                       actualbounds, grid)
end

Adapt.@adapt_structure WaterMedium

@inline get_template_grid(m::WaterMedium) = m.majorant_grid
@propagate_inbounds is_emissive(::WaterMedium) = false

"The water's tabulated spectrum `which` (0: absorption, 1: scattering) at the path's wavelengths."
@inline function waterspectrum(m::WaterMedium, spectra, λ::Wavelengths, which::Int32)
    v(l) = begin
        s = clamp((l - m.from) / m.step, 0f0, Float32(m.n - Int32(1)))
        i = min(unsafe_trunc(Int32, s), m.n - Int32(2))
        f = s - Float32(i)
        base = which * m.n
        @inbounds (1f0 - f) * spectra[base + i + 1] + f * spectra[base + i + 2]
    end
    return SpectralRadiance((v(λ.lambda[1]), v(λ.lambda[2]), v(λ.lambda[3]), v(λ.lambda[4])))
end

@propagate_inbounds function sample_point(m::WaterMedium, media, table::RGBToSpectrumTable, p::Point3f,
                                          λ::Wavelengths)::MediumProperties
    spectra = Raycore.deref(media, m.spectra)
    σ_a = waterspectrum(m, spectra, λ, Int32(0))
    σ_w = waterspectrum(m, spectra, λ, Int32(1))
    white = airscattering(media, m.air, p)
    # The air scatters without a preferred direction (the similarity relation); the water forward.
    g = σ_w[1] * m.g / (σ_w[1] + white + 1f-12)
    return MediumProperties(σ_a, σ_w + SpectralRadiance(white), SpectralRadiance(0f0), g)
end

@inline function control(m::WaterMedium, media, table::RGBToSpectrumTable, λ::Wavelengths)
    spectra = Raycore.deref(media, m.spectra)
    return MediumProperties(waterspectrum(m, spectra, λ, Int32(0)), waterspectrum(m, spectra, λ, Int32(1)),
                            SpectralRadiance(0f0), m.g)
end
@inline controlmajorant(m::WaterMedium) = m.base
# Air (eta=1) has no Snell window. Treating it as refracting rejected half
# of an environment's light samples while the escape MIS still counted them.
@inline refracting(m::WaterMedium) = m.ior > 1f0 ? m.ior : 0f0
@inline function medium_trace_distance(m::WaterMedium, media, p)
    s = airscattering(media, m.air, p)
    return s > 16f0 ? max(0.01f0, 4f0 / s) : Inf32
end

@propagate_inbounds residual(m::WaterMedium, media, table::RGBToSpectrumTable, p::Point3f, λ::Wavelengths)::MediumProperties =
    MediumProperties(SpectralRadiance(0f0), SpectralRadiance(airscattering(media, m.air, p)), SpectralRadiance(0f0), 0f0)

@propagate_inbounds function get_majorant(m::WaterMedium, table::RGBToSpectrumTable, ray::Raycore.Ray,
                                          t_min::Float32, t_max::Float32, λ::Wavelengths)::RayMajorantSegment
    top = 0f0
    if m.majorant_grid.voxels isa NativeMajorantValues
        top = m.majorant_grid.voxels.top
    else
        for v in m.majorant_grid.voxels
            top = max(top, v)
        end
    end
    return RayMajorantSegment(t_min, t_max, SpectralRadiance(top))
end

@propagate_inbounds function create_majorant_iterator(m::WaterMedium, table::RGBToSpectrumTable, ray::Raycore.Ray,
                                                      t_max::Float32, λ::Wavelengths)
    t_enter, t_exit = ray_bounds_intersect(ray.o, ray.d, m.bounds)
    t_enter = max(t_enter, 0f0)
    t_exit = min(t_exit, t_max)
    t_enter >= t_exit && return RayMajorantIterator(m.majorant_grid)
    dda = create_dda_iterator(m.majorant_grid, m.bounds, ray.o, ray.d, t_enter, t_exit, SpectralRadiance(1f0))
    return RayMajorantIterator(dda)
end

@propagate_inbounds create_majorant_iterator(m::WaterMedium, table::RGBToSpectrumTable, ray::Raycore.Ray, t_max::Float32,
                                             λ::Wavelengths, ::MajorantGrid) = create_majorant_iterator(m, table, ray, t_max, λ)

# The water control is spectral, but its air residual is achromatic. In ratio
# tracking the scalar majorant exponential cancels from every likelihood ratio:
# T *= (majorant-residual)/majorant, r_u does the same, and r_l stays one.
# Keeping that cancellation algebraic avoids four exponentials and spectral
# divisions per null collision, including their avoidable underflow. Other
# media retain the general spectral estimator.
@propagate_inbounds function transmittance_dda_helper(
    m::WaterMedium, table, ray, t_max, lambda, origin, dir, media, medium_idx, template_grid
)
    iter = create_majorant_iterator(m, table, ray, t_max, lambda, template_grid)
    mc = control(m, media, table, lambda)
    return water_ratio_tracking(iter, origin, dir, media, m, mc)
end

@propagate_inbounds function water_ratio_tracking(iter::RayMajorantIterator,
    origin::Point3f, dir::Vec3f, media, m::WaterMedium, mc::MediumProperties)
    one = SpectralRadiance(1f0)
    iter.mode == Int32(0) && return (one, one, one)
    transmission = exp(-(iter.t_max - iter.t_min) * σ_t(mc))
    residual_T = 1f0
    residual_pdf = 1f0
    rng = pcg32_init(pbrt_hash(origin), pbrt_hash(dir))
    current = iter
    for _ in Int32(1):majorant_segment_limit(iter)
        segment, current, valid = ray_majorant_next(current, media)
        valid || break
        bound = max(segment.σ_maj[1] - m.base, 0f0)
        bound < 1f-6 && continue
        t = segment.t_min
        while t < segment.t_max
            u, rng = pcg32_uniform_f32(rng)
            t += -log(max(1f-10, 1f0-u)) / bound
            t >= segment.t_max && break
            p = Point3f(origin + dir * t)
            weight = max(bound - airscattering(media,m.air,p),0f0) / bound
            residual_T *= weight
            residual_pdf *= weight
            estimate = transmission * (residual_T / (1f0 + residual_pdf))
            if max_component(estimate) < 0.05f0
                rr, rng = pcg32_uniform_f32(rng)
                rr < 0.75f0 && return (SpectralRadiance(0f0),SpectralRadiance(residual_pdf),one)
                residual_T *= 4f0
            end
            residual_T == 0f0 && return (SpectralRadiance(0f0),SpectralRadiance(residual_pdf),one)
        end
    end
    return (transmission * residual_T, SpectralRadiance(residual_pdf), one)
end
