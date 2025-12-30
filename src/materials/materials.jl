# Clean Material type definitions
# Each material is a data container with only its necessary parameters
# Integrators interpret these materials through shade() and sample_bounce()

# ============================================================================
# MatteMaterial - Lambertian diffuse with optional Oren-Nayar roughness
# ============================================================================

"""
    MatteMaterial(Kd::Texture, σ::Texture)

Matte (diffuse) material with Lambertian or Oren-Nayar BRDF.

* `Kd`: Spectral diffuse reflection (color texture)
* `σ`: Scalar roughness for Oren-Nayar model (0 = Lambertian)
"""
struct MatteMaterial{KdType, σType} <: Material
    Kd::Texture{RGBSpectrum, 2, KdType}
    σ::Texture{Float32, 2, σType}
end

function MatteMaterial(Kd::Texture, σ::Texture)
    KdType = typeof(Kd.data)
    σType = typeof(σ.data)
    MatteMaterial{KdType, σType}(Kd, σ)
end

# ============================================================================
# MirrorMaterial - Perfect specular reflection
# ============================================================================

"""
    MirrorMaterial(Kr::Texture)

Perfect mirror (specular reflection) material.

* `Kr`: Spectral reflectance (color texture)
"""
struct MirrorMaterial{KrType} <: Material
    Kr::Texture{RGBSpectrum, 2, KrType}
end

function MirrorMaterial(Kr::Texture)
    KrType = typeof(Kr.data)
    MirrorMaterial{KrType}(Kr)
end

# ============================================================================
# GlassMaterial - Specular reflection and transmission (glass/water)
# ============================================================================

"""
    GlassMaterial(Kr, Kt, u_roughness, v_roughness, index, remap_roughness)

Glass/dielectric material with reflection and transmission.

* `Kr`: Spectral reflectance
* `Kt`: Spectral transmittance
* `u_roughness`: Roughness in u direction (0 = perfect specular)
* `v_roughness`: Roughness in v direction (0 = perfect specular)
* `index`: Index of refraction
* `remap_roughness`: Whether to remap roughness to alpha
"""
struct GlassMaterial{KrType, KtType, RoughType, IndexType} <: Material
    Kr::Texture{RGBSpectrum, 2, KrType}
    Kt::Texture{RGBSpectrum, 2, KtType}
    u_roughness::Texture{Float32, 2, RoughType}
    v_roughness::Texture{Float32, 2, RoughType}
    index::Texture{Float32, 2, IndexType}
    remap_roughness::Bool
end

function GlassMaterial(
    Kr::Texture, Kt::Texture,
    u_roughness::Texture, v_roughness::Texture,
    index::Texture, remap_roughness::Bool
)
    KrType = typeof(Kr.data)
    KtType = typeof(Kt.data)
    RoughType = typeof(u_roughness.data)
    IndexType = typeof(index.data)
    GlassMaterial{KrType, KtType, RoughType, IndexType}(
        Kr, Kt, u_roughness, v_roughness, index, remap_roughness
    )
end

# ============================================================================
# PlasticMaterial - Diffuse + glossy specular (plastic-like appearance)
# ============================================================================

"""
    PlasticMaterial(Kd, Ks, roughness, remap_roughness)

Plastic material with diffuse and glossy specular components.

* `Kd`: Diffuse reflectance
* `Ks`: Specular reflectance
* `roughness`: Surface roughness
* `remap_roughness`: Whether to remap roughness to alpha
"""
struct PlasticMaterial{KdType, KsType, RoughType} <: Material
    Kd::Texture{RGBSpectrum, 2, KdType}
    Ks::Texture{RGBSpectrum, 2, KsType}
    roughness::Texture{Float32, 2, RoughType}
    remap_roughness::Bool
end

function PlasticMaterial(
    Kd::Texture, Ks::Texture,
    roughness::Texture, remap_roughness::Bool
)
    KdType = typeof(Kd.data)
    KsType = typeof(Ks.data)
    RoughType = typeof(roughness.data)
    PlasticMaterial{KdType, KsType, RoughType}(Kd, Ks, roughness, remap_roughness)
end

# ============================================================================
# BSDF computation for each material type
# ============================================================================

"""
Compute BSDF for MatteMaterial.
"""
@inline function compute_bsdf(m::MatteMaterial, si::SurfaceInteraction, ::Bool, transport)
    r = clamp(m.Kd(si))
    is_black(r) && return BSDF(si)
    σ = clamp(m.σ(si), 0f0, 90f0)
    lambertian = (σ ≈ 0.0f0)
    return BSDF(si, LambertianReflection(lambertian, r), OrenNayar(!lambertian, r, σ))
end

"""
Compute BSDF for MirrorMaterial.
"""
@inline function compute_bsdf(m::MirrorMaterial, si::SurfaceInteraction, ::Bool, transport)
    r = clamp(m.Kr(si))
    return BSDF(si, SpecularReflection(!is_black(r), r, FresnelNoOp()))
end

"""
Compute BSDF for GlassMaterial.
"""
@inline function compute_bsdf(g::GlassMaterial, si::SurfaceInteraction, allow_multiple_lobes::Bool, transport)
    η = g.index(si)
    u_roughness = g.u_roughness(si)
    v_roughness = g.v_roughness(si)

    r = clamp(g.Kr(si))
    t = clamp(g.Kt(si))
    r_black = is_black(r)
    t_black = is_black(t)
    r_black && t_black && return BSDF(si, η)

    is_specular = u_roughness ≈ 0f0 && v_roughness ≈ 0f0

    # For specular glass with multiple lobes, use FresnelSpecular
    if is_specular && allow_multiple_lobes
        return BSDF(si, η, FresnelSpecular(true, r, t, 1.0f0, η, transport))
    end

    if g.remap_roughness
        u_roughness = roughness_to_α(u_roughness)
        v_roughness = roughness_to_α(v_roughness)
    end

    # For rough glass, use FresnelMicrofacet which combines reflection and transmission
    if !is_specular && allow_multiple_lobes
        distribution = TrowbridgeReitzDistribution(u_roughness, v_roughness)
        return BSDF(si, η, FresnelMicrofacet(true, r, t, distribution, 1.0f0, η, transport))
    end

    # Fallback: separate BxDFs for specular-only or when multiple lobes not allowed
    distribution = is_specular ? TrowbridgeReitzDistribution() : TrowbridgeReitzDistribution(
        u_roughness, v_roughness,
    )
    fresnel = FresnelDielectric(1f0, η)
    return BSDF(
        si, η,
        SpecularReflection(!r_black && is_specular, r, fresnel),
        MicrofacetReflection(!r_black && !is_specular, r, distribution, fresnel, transport),
        SpecularTransmission(!t_black && is_specular, t, 1.0f0, η, transport),
        MicrofacetTransmission(!t_black && !is_specular, t, distribution, 1.0f0, η, transport)
    )
end

"""
Compute BSDF for PlasticMaterial.
"""
@inline function compute_bsdf(p::PlasticMaterial, si::SurfaceInteraction, ::Bool, transport)
    # Initialize diffuse component
    kd = clamp(p.Kd(si))
    bsdf_1 = LambertianReflection(!is_black(kd), kd)
    # Initialize specular component
    ks = clamp(p.Ks(si))
    is_black(ks) && return BSDF(si, bsdf_1)
    # Create microfacet distribution for plastic material
    fresnel = FresnelDielectric(1.5f0, 1f0)
    rough = p.roughness(si)
    p.remap_roughness && (rough = roughness_to_α(rough))
    distribution = TrowbridgeReitzDistribution(rough, rough)
    bsdf_2 = MicrofacetReflection(true, ks, distribution, fresnel, transport)
    return BSDF(si, bsdf_1, bsdf_2)
end

# ============================================================================
# Material Interface: shade() for each material type
# ============================================================================

# Non-allocating sum of le() over lights tuple (recursive for type stability)
@inline _sum_light_le(lights::Tuple{}, ray) = RGBSpectrum(0f0)
@inline function _sum_light_le(lights::Tuple, ray)
    return le(first(lights), ray) + _sum_light_le(Base.tail(lights), ray)
end

@inline function _shade_light(light, interaction, scene, hit_wo, bsdf, shading_n, beta)
    u_light = rand(Point2f)
    Li, wi, pdf, visibility = @inline sample_li(light, interaction, u_light, scene)
    (is_black(Li) || pdf ≈ 0f0) && return RGBSpectrum(0f0)
    f = @inline bsdf(hit_wo, wi)
    is_black(f) && return RGBSpectrum(0f0)
    !unoccluded(visibility, scene) && return RGBSpectrum(0f0)
    cos_theta = abs(wi ⋅ shading_n)
    return beta * f * Li * cos_theta / pdf
end

# Type-stable recursive light shading
@inline _shade_lights(::Tuple{}, interaction, scene, hit_wo, bsdf, shading_n, beta) = RGBSpectrum(0f0)
@inline function _shade_lights(lights::Tuple, interaction, scene, hit_wo, bsdf, shading_n, beta)
    first_contrib = _shade_light(first(lights), interaction, scene, hit_wo, bsdf, shading_n, beta)
    rest_contrib = _shade_lights(Base.tail(lights), interaction, scene, hit_wo, bsdf, shading_n, beta)
    return first_contrib + rest_contrib
end

"""
    shade(material::Material, ray, si, scene, beta, depth, max_depth) -> RGBSpectrum

Compute direct lighting and specular bounces for a surface hit.
This is the generic implementation that works for all material types.
"""
@inline function shade(material::Material, ray::RayDifferentials, si::SurfaceInteraction,
                       scene::S, beta::RGBSpectrum, depth::Int32, max_depth::Int32) where {S<:Scene}
    # Compute BSDF from material
    bsdf = compute_bsdf(material, si, false, Radiance)

    # Get hit info from SurfaceInteraction
    hit_p = si.core.p
    hit_wo = si.core.wo
    hit_n = si.core.n
    shading_n = si.shading.n

    # Compute direct lighting from all lights
    interaction = Interaction(hit_p, 0f0, hit_wo, hit_n)
    total = _shade_lights(scene.lights, interaction, scene, hit_wo, bsdf, shading_n, beta)

    # Handle specular bounces if we haven't exceeded max depth
    if depth < max_depth
        # Specular reflection
        total += _specular_bounce(Reflect(), bsdf, ray, si, scene, beta, depth, max_depth)
        # Specular transmission
        total += _specular_bounce(Transmit(), bsdf, ray, si, scene, beta, depth, max_depth)
    end

    return total
end

@inline _specular_type(::Reflect) = BSDF_REFLECTION | BSDF_SPECULAR
@inline _specular_type(::Transmit) = BSDF_TRANSMISSION | BSDF_SPECULAR

"""
Compute specular reflection or transmission contribution by tracing a bounce ray.
"""
@inline function _specular_bounce(type, bsdf::BSDF, ray::RayDifferentials, si::SurfaceInteraction,
                                  scene::S, beta::RGBSpectrum, depth::Int32, max_depth::Int32) where {S<:Scene}
    wo = si.core.wo
    ns = si.shading.n

    # Sample specular direction from BSDF
    u = rand(Point2f)
    wi, f, pdf, sampled_type = @inline sample_f(bsdf, wo, u, _specular_type(type))

    # Check for valid sample
    if !(pdf > 0f0 && !is_black(f) && abs(wi ⋅ ns) != 0f0)
        return RGBSpectrum(0f0)
    end

    # Compute throughput for this bounce
    bounce_beta = beta * f * abs(wi ⋅ ns) / pdf

    # Spawn bounce ray
    bounce_ray = RayDifferentials(spawn_ray(si, wi))

    # Add ray differentials if available
    if ray.has_differentials
        is_reflection = (sampled_type & BSDF_REFLECTION) != 0
        diff_type = is_reflection ? Reflect() : Transmit()
        bounce_ray = _specular_differentials(diff_type, bounce_ray, bsdf, si, ray, wo, wi)
    end

    # Trace bounce ray through scene
    hit, primitive, bounce_si = intersect!(scene, bounce_ray)

    if !hit
        result = _sum_light_le(scene.lights, bounce_ray)
        return bounce_beta * result
    end

    # Recursively shade the hit point
    return shade_material(
        scene.aggregate.materials, primitive.metadata,
        bounce_ray, bounce_si, scene, bounce_beta, depth + Int32(1), max_depth
    )
end

"""
Compute ray differentials for specular reflection.
"""
@inline function _specular_differentials(::Reflect, rd, bsdf, si, ray, wo, wi)
    ns = si.shading.n
    rx_origin = si.core.p + si.∂p∂x
    ry_origin = si.core.p + si.∂p∂y
    ∂n∂x = si.shading.∂n∂u * si.∂u∂x + si.shading.∂n∂v * si.∂v∂x
    ∂n∂y = si.shading.∂n∂u * si.∂u∂y + si.shading.∂n∂v * si.∂v∂y
    ∂wo∂x = -ray.rx_direction - wo
    ∂wo∂y = -ray.ry_direction - wo
    ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
    ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
    rx_direction = wi - ∂wo∂x + 2f0 * (wo ⋅ ns) * ∂n∂x + ∂dn∂x * ns
    ry_direction = wi - ∂wo∂y + 2f0 * (wo ⋅ ns) * ∂n∂y + ∂dn∂y * ns
    return RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
end

"""
Compute ray differentials for specular transmission.
"""
@inline function _specular_differentials(::Transmit, rd, bsdf, si, ray, wo, wi)
    ns = si.shading.n
    rx_origin = si.core.p + si.∂p∂x
    ry_origin = si.core.p + si.∂p∂y
    ∂n∂x = si.shading.∂n∂u * si.∂u∂x + si.shading.∂n∂v * si.∂v∂x
    ∂n∂y = si.shading.∂n∂u * si.∂u∂y + si.shading.∂n∂v * si.∂v∂y
    η = 1f0 / bsdf.η
    if (wo ⋅ ns) < 0f0
        η = 1f0 / η
        ∂n∂x, ∂n∂y, ns = -∂n∂x, -∂n∂y, -ns
    end
    ∂wo∂x = -ray.rx_direction - wo
    ∂wo∂y = -ray.ry_direction - wo
    ∂dn∂x = ∂wo∂x ⋅ ns + wo ⋅ ∂n∂x
    ∂dn∂y = ∂wo∂y ⋅ ns + wo ⋅ ∂n∂y
    μ = η * (wo ⋅ ns) - abs(wi ⋅ ns)
    ν = η - (η * η * (wo ⋅ ns)) / abs(wi ⋅ ns)
    ∂μ∂x = ν * ∂dn∂x
    ∂μ∂y = ν * ∂dn∂y
    rx_direction = wi - η * ∂wo∂x + μ * ∂n∂x + ∂μ∂x * ns
    ry_direction = wi - η * ∂wo∂y + μ * ∂n∂y + ∂μ∂y * ns
    return RayDifferentials(rd, rx_origin=rx_origin, ry_origin=ry_origin, rx_direction=rx_direction, ry_direction=ry_direction)
end

# ============================================================================
# sample_bounce for path tracing
# ============================================================================

"""
    sample_bounce(material::Material, ray, si, scene, beta, depth) -> (should_bounce, new_ray, new_beta, new_depth)

Sample BSDF to generate a bounce ray for path continuation.
"""
@inline function sample_bounce(material::Material, ray::RayDifferentials, si::SurfaceInteraction,
                               scene::Scene, beta::RGBSpectrum, depth::Int32)
    # Compute BSDF
    bsdf = compute_bsdf(material, si, true, Radiance)

    hit_wo = si.core.wo
    shading_n = si.shading.n

    # Sample BSDF for bounce direction
    u_bsdf = rand(Point2f)
    wi, f, pdf, sampled_type = sample_f(bsdf, hit_wo, u_bsdf, BSDF_ALL)

    # Check if valid sample
    if pdf > 0f0 && !is_black(f)
        # Update throughput
        cos_theta = abs(wi ⋅ shading_n)
        new_beta = beta * f * cos_theta / pdf

        # Russian roulette for path termination (after first few bounces)
        terminate = false
        if depth > Int32(2)
            q = max(0.05f0, 1f0 - to_Y(new_beta))
            if rand(Float32) < q
                terminate = true
            else
                new_beta = new_beta / (1f0 - q)
            end
        end

        if !terminate
            # Spawn bounce ray
            bounce_ray = RayDifferentials(spawn_ray(si, wi))
            return (true, bounce_ray, new_beta, depth + Int32(1))
        end
    end

    # No valid bounce
    dummy_ray = RayDifferentials(Ray(Point3f(0), Vec3f(0, 0, 1), Inf32, 0f0))
    return (false, dummy_ray, RGBSpectrum(0f0), Int32(0))
end
