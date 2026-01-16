# Light Sampler - Importance sampling of lights for direct lighting
# Following pbrt-v4's LightSampler abstraction
#
# Available samplers:
# - UniformLightSampler: O(1) uniform random selection (baseline)
# - PowerLightSampler: O(1) power-weighted selection using alias table
#
# The sampler provides:
# - sample(sampler, u) -> (light_idx, pmf): Sample a light with probability proportional to importance
# - pmf(sampler, light_idx) -> Float32: Get PMF for a specific light

# ============================================================================
# Alias Table - O(1) Weighted Discrete Sampling
# ============================================================================

"""
    AliasTable

Walker's alias method for O(1) sampling from a discrete distribution.
Each bin stores:
- `p`: The PMF (probability) for this index
- `q`: The threshold for choosing this index vs the alias
- `alias`: The aliased index if u > q

Following pbrt-v4's implementation in util/sampling.h
"""
struct AliasTable
    # Packed bins: (p, q, alias) for each entry
    # p = actual PMF for this index
    # q = threshold for choosing this vs alias
    # alias = alternative index if u > q
    p::Vector{Float32}
    q::Vector{Float32}
    alias::Vector{Int32}
end

"""
    AliasTable(weights::AbstractVector{Float32})

Construct an alias table from weights (need not sum to 1).
"""
function AliasTable(weights::AbstractVector{<:Real})
    n = length(weights)
    if n == 0
        return AliasTable(Float32[], Float32[], Int32[])
    end

    # Normalize to get PMFs
    total = sum(weights)
    if total <= 0
        # All zero weights - uniform distribution
        p = fill(1f0 / n, n)
    else
        p = Float32[w / total for w in weights]
    end

    q = Vector{Float32}(undef, n)
    alias = Vector{Int32}(undef, n)

    # Create work lists
    # "under" contains indices with p_hat < 1
    # "over" contains indices with p_hat >= 1
    under = Int32[]
    over = Int32[]
    p_hat = Vector{Float32}(undef, n)

    for i in 1:n
        p_hat[i] = p[i] * n
        if p_hat[i] < 1f0
            push!(under, Int32(i))
        else
            push!(over, Int32(i))
        end
    end

    # Process under and over together
    while !isempty(under) && !isempty(over)
        un = pop!(under)
        ov = pop!(over)

        # Set threshold and alias for under-probability bin
        q[un] = p_hat[un]
        alias[un] = ov

        # Transfer excess probability to over bin
        p_excess = p_hat[un] + p_hat[ov] - 1f0
        p_hat[ov] = p_excess

        if p_excess < 1f0
            push!(under, ov)
        else
            push!(over, ov)
        end
    end

    # Handle remaining items (numerical precision may leave some)
    while !isempty(over)
        ov = pop!(over)
        q[ov] = 1f0
        alias[ov] = Int32(-1)  # Won't be used
    end
    while !isempty(under)
        un = pop!(under)
        q[un] = 1f0
        alias[un] = Int32(-1)  # Won't be used
    end

    return AliasTable(p, q, alias)
end

"""
    sample(table::AliasTable, u::Float32) -> (index::Int32, pmf::Float32)

Sample from the alias table using uniform random `u ∈ [0,1)`.
Returns 1-based index and its PMF.
"""
@propagate_inbounds function sample(table::AliasTable, u::Float32)
    n = length(table.p)
    if n == 0
        return (Int32(0), 0f0)
    end

    # Scale u to [0, n) and extract integer and fractional parts
    scaled = u * Float32(n)
    offset = min(floor(Int32, scaled), Int32(n - 1))
    up = min(scaled - Float32(offset), 0.99999994f0)  # OneMinusEpsilon

    # 1-based index
    idx = offset + Int32(1)

    # Choose between original and alias based on threshold
    if up < table.q[idx]
        return (idx, table.p[idx])
    else
        alias_idx = table.alias[idx]
        if alias_idx < 1
            # Fallback - shouldn't happen with proper construction
            return (idx, table.p[idx])
        end
        return (alias_idx, table.p[alias_idx])
    end
end

"""
    pmf(table::AliasTable, idx::Int32) -> Float32

Get the PMF for index `idx` (1-based).
"""
@propagate_inbounds function pmf(table::AliasTable, idx::Int32)
    if idx < 1 || idx > length(table.p)
        return 0f0
    end
    return table.p[idx]
end

Base.length(table::AliasTable) = length(table.p)
Base.isempty(table::AliasTable) = isempty(table.p)

# ============================================================================
# Light Sampler Abstract Type
# ============================================================================

abstract type LightSampler end

# ============================================================================
# Uniform Light Sampler
# ============================================================================

"""
    UniformLightSampler

Simple uniform light selection. O(1) but ignores light importance.
This is the baseline sampler equivalent to the current Hikari behavior.
"""
struct UniformLightSampler <: LightSampler
    num_lights::Int32
end

UniformLightSampler(lights::Tuple) = UniformLightSampler(Int32(length(lights)))
UniformLightSampler(num_lights::Integer) = UniformLightSampler(Int32(num_lights))

"""
    sample(sampler::UniformLightSampler, u::Float32) -> (light_idx::Int32, pmf::Float32)

Sample a light uniformly. Returns 1-based index and PMF.
"""
@propagate_inbounds function sample(sampler::UniformLightSampler, u::Float32)
    n = sampler.num_lights
    if n < Int32(1)
        return (Int32(0), 0f0)
    end
    idx = min(floor(Int32, u * Float32(n)), n - Int32(1)) + Int32(1)
    return (idx, 1f0 / Float32(n))
end

"""
    pmf(sampler::UniformLightSampler, light_idx::Int32) -> Float32

PMF for any light is 1/N.
"""
@propagate_inbounds function pmf(sampler::UniformLightSampler, ::Int32)
    n = sampler.num_lights
    n < Int32(1) ? 0f0 : 1f0 / Float32(n)
end

# ============================================================================
# Power Light Sampler
# ============================================================================

"""
    PowerLightSampler

Sample lights proportional to their total power using an alias table.
O(1) sampling with O(N) construction. Good for scenes with varying light intensities.

Following pbrt-v4's PowerLightSampler which uses light.Phi() to estimate power.
"""
struct PowerLightSampler <: LightSampler
    alias_table::AliasTable
end

"""
    PowerLightSampler(lights::Tuple; scene_radius::Float32=10f0)

Construct a power-based light sampler from a tuple of lights.
Estimates light power using luminance of the light's intensity/radiance.

For infinite lights (DirectionalLight, EnvironmentLight), the scene_radius
is needed to compute meaningful power estimates following pbrt-v4's approach.

# Arguments
- `lights::Tuple`: Tuple of light sources
- `scene_radius::Float32`: Radius of the scene's bounding sphere (default: 10.0)
"""
function PowerLightSampler(lights::Tuple; scene_radius::Float32=10f0)
    n = length(lights)
    if n == 0
        return PowerLightSampler(AliasTable(Float32[]))
    end

    # Compute power estimate for each light
    powers = Vector{Float32}(undef, n)
    for i in 1:n
        powers[i] = estimate_light_power(lights[i], scene_radius)
    end

    # If all powers are zero, fall back to uniform
    total_power = sum(powers)
    if total_power <= 0f0
        fill!(powers, 1f0)
    end

    return PowerLightSampler(AliasTable(powers))
end

"""
    sample(sampler::PowerLightSampler, u::Float32) -> (light_idx::Int32, pmf::Float32)

Sample a light with probability proportional to power.
"""
@propagate_inbounds function sample(sampler::PowerLightSampler, u::Float32)
    sample(sampler.alias_table, u)
end

"""
    pmf(sampler::PowerLightSampler, light_idx::Int32) -> Float32

Get PMF for a specific light index.
"""
@propagate_inbounds function pmf(sampler::PowerLightSampler, light_idx::Int32)
    pmf(sampler.alias_table, light_idx)
end

# ============================================================================
# Light Power Estimation
# ============================================================================

"""
    estimate_light_power(light, scene_radius::Float32) -> Float32

Estimate the total power (flux) of a light for importance sampling.
Following pbrt-v4's Light::Phi() which returns total emitted power.

For point/spot lights: Phi = 4π * I (or 2π * I * cone_factor for spot)
For directional lights: Phi = π * sceneRadius² * I
For environment lights: Phi = 4π² * sceneRadius² * average_radiance

The scene_radius is required for infinite lights to compute meaningful power.
"""
@propagate_inbounds estimate_light_power(light::Light, ::Float32) = 1f0  # Fallback

@propagate_inbounds function estimate_light_power(light::PointLight, ::Float32)
    # pbrt-v4: 4 * Pi * scale * I->Sample(lambda)
    # Total power = 4π * intensity
    4f0 * Float32(π) * luminance(light.i)
end

@propagate_inbounds function estimate_light_power(light::SpotLight, ::Float32)
    # pbrt-v4: scale * Iemit * 2 * Pi * ((1 - cosFalloffStart) + (cosFalloffStart - cosFalloffEnd) / 2)
    # Hikari stores: cos_falloff_start and cos_total_width (= cosFalloffEnd)
    # Formula simplifies to: 2π * I * (1 - 0.5*(cosFalloffStart + cosFalloffEnd))
    cone_factor = 1f0 - 0.5f0 * (light.cos_falloff_start + light.cos_total_width)
    2f0 * Float32(π) * luminance(light.i) * cone_factor
end

@propagate_inbounds function estimate_light_power(light::DirectionalLight, scene_radius::Float32)
    # pbrt-v4: scale * Lemit * Pi * Sqr(sceneRadius)
    Float32(π) * scene_radius^2 * luminance(light.i)
end

@propagate_inbounds function estimate_light_power(light::SunLight, scene_radius::Float32)
    # Similar to directional light
    Float32(π) * scene_radius^2 * luminance(light.l)
end

@propagate_inbounds function estimate_light_power(light::SunSkyLight, scene_radius::Float32)
    # Sun contributes most of the direct illumination
    Float32(π) * scene_radius^2 * luminance(light.sun_intensity)
end

@propagate_inbounds function estimate_light_power(light::AmbientLight, ::Float32)
    # Ambient lights provide uniform illumination - less important for direct lighting
    # Use small weight since they don't contribute to direct lighting variance reduction
    4f0 * Float32(π) * luminance(light.i) * 0.1f0
end

# Helper to get total distribution integral (dispatch on distribution type)
distribution_integral(d::Distribution2D) = d.p_marginal.func_int
distribution_integral(d::FlatDistribution2D) = d.marginal_func_int

@propagate_inbounds function estimate_light_power(light::EnvironmentLight, scene_radius::Float32)
    # pbrt-v4: 4 * Pi * Pi * Sqr(sceneRadius) * scale * sumL / (width * height)
    # The distribution integral gives us sum of luminance-weighted pixels
    # We approximate average radiance from this
    total = distribution_integral(light.env_map.distribution)
    # Power = 4π² * r² * average_radiance * scale
    4f0 * Float32(π)^2 * scene_radius^2 * Float32(total) * luminance(light.scale)
end

"""
    luminance(rgb) -> Float32

Compute luminance from RGB using standard coefficients (Rec. 709).
"""
@propagate_inbounds function luminance(rgb::RGB)
    0.212671f0 * Float32(rgb.r) + 0.715160f0 * Float32(rgb.g) + 0.072169f0 * Float32(rgb.b)
end

@propagate_inbounds function luminance(s::RGBSpectrum)
    0.212671f0 * s.c[1] + 0.715160f0 * s.c[2] + 0.072169f0 * s.c[3]
end

# ============================================================================
# GPU-Compatible Light Sampler (Pre-computed Arrays)
# ============================================================================

"""
    LightSamplerData

GPU-compatible light sampler data structure.
Stores alias table as flat arrays that can be uploaded to GPU.
"""
struct LightSamplerData
    p::Vector{Float32}      # PMF values
    q::Vector{Float32}      # Alias thresholds
    alias::Vector{Int32}    # Alias indices
    num_lights::Int32
end

function LightSamplerData(sampler::PowerLightSampler)
    table = sampler.alias_table
    LightSamplerData(
        copy(table.p),
        copy(table.q),
        copy(table.alias),
        Int32(length(table.p))
    )
end

function LightSamplerData(sampler::UniformLightSampler)
    n = sampler.num_lights
    if n < 1
        return LightSamplerData(Float32[], Float32[], Int32[], Int32(0))
    end
    p = fill(1f0 / Float32(n), n)
    q = fill(1f0, n)
    alias = fill(Int32(-1), n)
    LightSamplerData(p, q, alias, n)
end

"""
    sample_light_sampler(p, q, alias, u::Float32) -> (light_idx::Int32, pmf::Float32)

GPU-compatible light sampling using pre-computed alias table arrays.
"""
@propagate_inbounds function sample_light_sampler(
    p::AbstractVector{Float32},
    q::AbstractVector{Float32},
    alias::AbstractVector{Int32},
    u::Float32
)
    n = length(p)
    if n == 0
        return (Int32(0), 0f0)
    end

    # Scale u to [0, n) and extract integer and fractional parts
    scaled = u * Float32(n)
    offset = min(floor(Int32, scaled), Int32(n - 1))
    up = min(scaled - Float32(offset), 0.99999994f0)

    # 1-based index
    idx = offset + Int32(1)

    # Choose between original and alias based on threshold
    if up < q[idx]
        return (idx, p[idx])
    else
        alias_idx = alias[idx]
        if alias_idx < 1
            return (idx, p[idx])
        end
        return (alias_idx, p[alias_idx])
    end
end

# ============================================================================
# Light Sampler Factory
# ============================================================================

"""
    create_light_sampler(lights::Tuple; method::Symbol=:power, scene_radius::Float32=10f0) -> LightSampler

Create a light sampler for the given lights.

# Arguments
- `lights::Tuple`: Tuple of light sources
- `method::Symbol`: Sampling method (`:uniform` or `:power`)
- `scene_radius::Float32`: Scene bounding sphere radius (for power-weighted sampling of infinite lights)

# Methods
- `:uniform`: Uniform random selection (baseline)
- `:power`: Power-weighted selection (recommended for varying light intensities)
"""
function create_light_sampler(lights::Tuple; method::Symbol=:power, scene_radius::Float32=10f0)
    if method == :uniform
        return UniformLightSampler(lights)
    elseif method == :power
        return PowerLightSampler(lights; scene_radius=scene_radius)
    else
        error("Unknown light sampler method: $method")
    end
end
