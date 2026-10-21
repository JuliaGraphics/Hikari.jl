"""
    NanoGridMedium(density::NanoTree; temperature=nothing, σ_a=RGBSpectrum(0),
                   σ_s=RGBSpectrum(1), g=0, emission_scale=1,
                   temperature_offset=0, temperature_scale=1, majorant_res=(64,64,64))

Smoke, clouds, or hot absorbing matter on device-built NanoVDB fields. `density`
multiplies the absorption and scattering coefficients (per metre); `temperature` is
an optional independently positioned NanoTree in kelvin. Its thermal source radiance
is Planck's law in W/(m² sr nm), multiplied by `emission_scale`. The integrator
multiplies this source by local absorption, as required by Kirchhoff's law. Density
zero therefore emits nothing. This is a thermal continuum model, not a combustion
solver or a model of spectral line emission.

Temperature is interpreted as `(value - temperature_offset) * temperature_scale`.
Unlike pbrt's peak-normalized blackbody, the physical source gets brighter with
temperature. A separate grid is optional for non-emissive storm clouds.
"""
struct NanoGridMedium{D<:NanoTree,T,M<:MajorantGrid} <: Medium
    density::D
    temperature::T
    σ_a::RGBSpectrum
    σ_s::RGBSpectrum
    g::Float32
    emission_scale::Float32
    temperature_offset::Float32
    temperature_scale::Float32
    bounds::Bounds3
    majorant_grid::M
    max_density::Float32
end
Adapt.@adapt_structure NanoGridMedium

function NanoGridMedium(density::NanoTree; temperature::Union{NanoTree,Nothing}=nothing,
                        σ_a::RGBSpectrum=RGBSpectrum(0f0), σ_s::RGBSpectrum=RGBSpectrum(1f0),
                        g::Real=0, emission_scale::Real=1, temperature_offset::Real=0,
                        temperature_scale::Real=1, majorant_res=(64, 64, 64))
    res = Vec{3,Int32}(majorant_res...)
    all(res .> 0) || throw(ArgumentError("majorant resolution must be positive"))
    abs(g) < 1 || throw(ArgumentError("phase asymmetry must lie strictly between -1 and 1"))
    isfinite(emission_scale) && emission_scale >= 0 || throw(ArgumentError("emission scale must be finite and nonnegative"))
    # Interpolation extends half a voxel beyond the leaf boxes. Enclose its full
    # support, including the blend to background at the outermost voxel centres.
    lo = density.lo + (Vec3f(density.imin) .- 0.5f0) * density.voxel
    hi = density.lo + (Vec3f(density.imax) .+ 0.5f0) * density.voxel
    bounds = Bounds3(Point3f(lo), Point3f(hi))
    grid = nanomajorant(density, bounds, res)
    return NanoGridMedium(density, temperature, σ_a, σ_s, Float32(g), Float32(emission_scale),
                          Float32(temperature_offset), Float32(temperature_scale), bounds, grid,
                          maximum(grid.voxels))
end

"Planck spectral radiance per nm, computed in micrometres to avoid Float32 underflow."
@inline function thermalradiance(λ::Float32, temperature::Float32)
    temperature > 0f0 || return 0f0
    um = λ * 1f-3
    x = 1.4387769f4 / (um * temperature)
    x < 80f0 || return 0f0
    return 1.1910429f5 / (um^5 * (exp(x) - 1f0))
end

@inline thermal_source(::Nothing, media, p, λ, offset, scale) = SpectralRadiance(0f0)
@inline function thermal_source(t::NanoTree, media, p, λ, offset, scale)
    temperature = (nanosample(t, Raycore.deref(media, t.words), p) - offset) * scale
    return SpectralRadiance(ntuple(i -> thermalradiance(λ.lambda[i], temperature), Val(4)))
end

@inline is_emissive(m::NanoGridMedium) = m.temperature !== nothing && m.emission_scale > 0f0
@inline get_template_grid(m::NanoGridMedium) = m.majorant_grid

@propagate_inbounds function sample_point(m::NanoGridMedium, media, table::RGBToSpectrumTable,
                                         p::Point3f, λ::Wavelengths)::MediumProperties
    d = max(0f0, nanosample(m.density, Raycore.deref(media, m.density.words), p))
    σa = uplift_rgb_unbounded(table, m.σ_a, λ) * d
    σs = uplift_rgb_unbounded(table, m.σ_s, λ) * d
    Le = m.emission_scale * thermal_source(m.temperature, media, p, λ, m.temperature_offset, m.temperature_scale)
    return MediumProperties(σa, σs, Le, m.g)
end

@propagate_inbounds function get_majorant(m::NanoGridMedium, table::RGBToSpectrumTable, ray::Raycore.Ray,
                                         t_min::Float32, t_max::Float32, λ::Wavelengths)
    σt = uplift_rgb_unbounded(table, m.σ_a, λ) + uplift_rgb_unbounded(table, m.σ_s, λ)
    return RayMajorantSegment(t_min, t_max, m.max_density * σt)
end

@propagate_inbounds function create_majorant_iterator(m::NanoGridMedium, table::RGBToSpectrumTable,
                                                     ray::Raycore.Ray, t_max::Float32, λ::Wavelengths)
    enter, leave = ray_bounds_intersect(ray.o, ray.d, m.bounds)
    enter = max(enter, 0f0); leave = min(leave, t_max)
    enter >= leave && return RayMajorantIterator(m.majorant_grid)
    σt = uplift_rgb_unbounded(table, m.σ_a, λ) + uplift_rgb_unbounded(table, m.σ_s, λ)
    return RayMajorantIterator(create_dda_iterator(m.majorant_grid, m.bounds, ray.o, ray.d, enter, leave, σt))
end

@propagate_inbounds create_majorant_iterator(m::NanoGridMedium, table::RGBToSpectrumTable, ray::Raycore.Ray,
                                             t_max::Float32, λ::Wavelengths, ::MajorantGrid) =
    create_majorant_iterator(m, table, ray, t_max, λ)
