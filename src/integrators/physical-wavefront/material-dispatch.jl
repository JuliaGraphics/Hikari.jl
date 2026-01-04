# Material dispatch for PhysicalWavefront path tracing
# Uses @generated functions for type-stable dispatch over Hikari's material tuple
# Each material type implements sample_bsdf_spectral() from spectral-eval.jl

# ============================================================================
# Spectral Material Sampling Dispatch
# ============================================================================

"""
    sample_spectral_material(materials::NTuple{N}, idx::MaterialIndex, wo, ns, uv, lambda, u, rng)

Type-stable dispatch for spectral BSDF sampling over Hikari's material tuple.
Returns SpectralBSDFSample from the appropriate material type.

Uses @generated to create efficient branching code at compile time.
"""
@inline @generated function sample_spectral_material(
    materials::NTuple{N,Any}, idx::MaterialIndex,
    wo::Vec3f, ns::Vec3f, uv::Point2f,
    lambda::Wavelengths, u::Point2f, rng::Float32
) where {N}
    branches = [quote
        @inbounds if idx.material_type === UInt8($i)
            return @inline sample_bsdf_spectral(materials[$i][idx.material_idx], wo, ns, uv, lambda, u, rng)
        end
    end for i in 1:N]

    # GPU-compatible fallback (no error/throw)
    quote
        $(branches...)
        return SpectralBSDFSample()
    end
end

# ============================================================================
# Spectral BSDF Evaluation Dispatch
# ============================================================================

"""
    evaluate_spectral_material(materials::NTuple{N}, idx::MaterialIndex, wo, wi, ns, uv, lambda)

Type-stable dispatch for spectral BSDF evaluation.
Returns (f::SpectralRadiance, pdf::Float32).
"""
@inline @generated function evaluate_spectral_material(
    materials::NTuple{N,Any}, idx::MaterialIndex,
    wo::Vec3f, wi::Vec3f, ns::Vec3f, uv::Point2f,
    lambda::Wavelengths
) where {N}
    branches = [quote
        @inbounds if idx.material_type === UInt8($i)
            return @inline evaluate_bsdf_spectral(materials[$i][idx.material_idx], wo, wi, ns, uv, lambda)
        end
    end for i in 1:N]

    quote
        $(branches...)
        return (SpectralRadiance(0f0), 0f0)
    end
end

# ============================================================================
# Emission Dispatch
# ============================================================================

"""
    get_emission_spectral_dispatch(materials::NTuple{N}, idx::MaterialIndex, wo, n, uv, lambda)

Type-stable dispatch for getting spectral emission from materials.
Returns SpectralRadiance (zero for non-emissive materials).
"""
@inline @generated function get_emission_spectral_dispatch(
    materials::NTuple{N,Any}, idx::MaterialIndex,
    wo::Vec3f, n::Vec3f, uv::Point2f, lambda::Wavelengths
) where {N}
    branches = [quote
        @inbounds if idx.material_type === UInt8($i)
            return @inline get_emission_spectral(materials[$i][idx.material_idx], wo, n, uv, lambda)
        end
    end for i in 1:N]

    quote
        $(branches...)
        return SpectralRadiance(0f0)
    end
end

"""
    get_emission_spectral_uv_dispatch(materials::NTuple{N}, idx::MaterialIndex, uv, lambda)

Type-stable dispatch for getting spectral emission without directional check.
Returns SpectralRadiance (zero for non-emissive materials).
"""
@inline @generated function get_emission_spectral_uv_dispatch(
    materials::NTuple{N,Any}, idx::MaterialIndex,
    uv::Point2f, lambda::Wavelengths
) where {N}
    branches = [quote
        @inbounds if idx.material_type === UInt8($i)
            return @inline get_emission_spectral(materials[$i][idx.material_idx], uv, lambda)
        end
    end for i in 1:N]

    quote
        $(branches...)
        return SpectralRadiance(0f0)
    end
end

# ============================================================================
# Emissive Check Dispatch
# ============================================================================

"""
    is_emissive_dispatch(materials::NTuple{N}, idx::MaterialIndex)

Type-stable dispatch for checking if a material is emissive.
Returns Bool.
"""
@inline @generated function is_emissive_dispatch(
    materials::NTuple{N,Any}, idx::MaterialIndex
) where {N}
    branches = [quote
        @inbounds if idx.material_type === UInt8($i)
            return @inline is_emissive(materials[$i][idx.material_idx])
        end
    end for i in 1:N]

    quote
        $(branches...)
        return false
    end
end

# ============================================================================
# Albedo Dispatch (for denoising auxiliary buffers)
# ============================================================================

"""
    get_albedo_spectral_dispatch(materials::NTuple{N}, idx::MaterialIndex, uv, lambda)

Type-stable dispatch for getting material albedo for denoising.
Returns SpectralRadiance.
"""
@inline @generated function get_albedo_spectral_dispatch(
    materials::NTuple{N,Any}, idx::MaterialIndex,
    uv::Point2f, lambda::Wavelengths
) where {N}
    branches = [quote
        @inbounds if idx.material_type === UInt8($i)
            return @inline get_albedo_spectral(materials[$i][idx.material_idx], uv, lambda)
        end
    end for i in 1:N]

    quote
        $(branches...)
        return SpectralRadiance(0.5f0)  # Fallback grey
    end
end

# ============================================================================
# Combined Material Evaluation for Wavefront Pipeline
# ============================================================================

"""
    PWMaterialEvalResult

Result of evaluating a material for the wavefront pipeline.
"""
struct PWMaterialEvalResult
    # BSDF sample result
    sample::SpectralBSDFSample

    # Emission (if material is emissive)
    Le::SpectralRadiance

    # Material properties for path decisions
    is_emissive::Bool
end

@inline function PWMaterialEvalResult()
    return PWMaterialEvalResult(
        SpectralBSDFSample(),
        SpectralRadiance(0f0),
        false
    )
end

"""
    evaluate_material_complete(materials, idx, wo, ns, n, uv, lambda, u, rng)

Complete material evaluation for wavefront pipeline.
Returns PWMaterialEvalResult with BSDF sample and emission.
"""
@inline @generated function evaluate_material_complete(
    materials::NTuple{N,Any}, idx::MaterialIndex,
    wo::Vec3f, ns::Vec3f, n::Vec3f, uv::Point2f,
    lambda::Wavelengths, u::Point2f, rng::Float32
) where {N}
    branches = [quote
        @inbounds if idx.material_type === UInt8($i)
            sample = @inline sample_bsdf_spectral(materials[$i][idx.material_idx], wo, ns, uv, lambda, u, rng)
            Le = @inline get_emission_spectral(materials[$i][idx.material_idx], wo, n, uv, lambda)
            is_em = @inline is_emissive(materials[$i][idx.material_idx])
            return PWMaterialEvalResult(sample, Le, is_em)
        end
    end for i in 1:N]

    quote
        $(branches...)
        return PWMaterialEvalResult()
    end
end

# ============================================================================
# Spawn Indirect Ray
# ============================================================================

"""
    spawn_spectral_ray(item::PWMaterialEvalWorkItem, wi::Vec3f, new_beta::SpectralRadiance,
                       is_specular::Bool, pdf::Float32, eta_scale::Float32) -> PWRayWorkItem

Create a new ray work item for indirect lighting from a material evaluation.
"""
@inline function spawn_spectral_ray(
    item::PWMaterialEvalWorkItem,
    wi::Vec3f,
    new_beta::SpectralRadiance,
    is_specular::Bool,
    pdf::Float32,
    eta_scale::Float32
)
    # Offset origin slightly along normal to avoid self-intersection
    offset = 1f-4 * item.n

    # Determine offset direction based on wi direction relative to normal
    ray_origin = if dot(wi, item.n) > 0f0
        Point3f((item.pi + offset)...)
    else
        Point3f((item.pi - offset)...)
    end

    ray = Raycore.Ray(o=ray_origin, d=wi)

    # Update path state
    new_depth = item.depth + Int32(1)
    any_non_specular = item.any_non_specular_bounces || !is_specular

    # Update r_u (unidirectional PDF) for non-specular bounces
    new_r_u = if is_specular
        item.r_u
    else
        item.r_u * SpectralRadiance(pdf)
    end

    # Create light sample context for MIS
    prev_ctx = PWLightSampleContext(item.pi, item.n, item.ns)

    return PWRayWorkItem(
        ray,
        new_depth,
        item.lambda,
        item.pixel_index,
        new_beta,
        new_r_u,
        SpectralRadiance(1f0),  # r_l reset for new path segment
        prev_ctx,
        eta_scale,
        is_specular,
        any_non_specular
    )
end

# ============================================================================
# Russian Roulette
# ============================================================================

"""
    russian_roulette_spectral(beta::SpectralRadiance, depth::Int32, rr_sample::Float32, min_depth::Int32=3)

Apply Russian roulette for path termination.
Returns (should_continue::Bool, new_beta::SpectralRadiance).
"""
@inline function russian_roulette_spectral(
    beta::SpectralRadiance,
    depth::Int32,
    rr_sample::Float32,
    min_depth::Int32=Int32(3)
)
    # Don't apply RR for first few bounces
    if depth <= min_depth
        return (true, beta)
    end

    # Use max component of throughput for continuation probability
    # Higher throughput = higher probability of continuation
    max_comp = max_component(beta)
    q = max(0.05f0, 1f0 - max_comp)

    if rr_sample < q
        # Terminate path
        return (false, beta)
    else
        # Continue with boosted beta
        scale = 1f0 / (1f0 - q)
        return (true, beta * scale)
    end
end
