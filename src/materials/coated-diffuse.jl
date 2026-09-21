# ============================================================================
# CoatedDiffuse - Layered material with dielectric coating over diffuse
# ============================================================================
# Port of pbrt-v4's CoatedDiffuse using LayeredBxDF
#
# The material consists of:
# - Top layer: Dielectric interface (can be rough or smooth)
# - Bottom layer: Diffuse reflector
# - Optional absorbing medium between layers
#
# Reference: pbrt-v4 src/pbrt/bxdfs.h LayeredBxDF, CoatedDiffuseBxDF

"""
    CoatedDiffuse

A layered material with a dielectric coating over a diffuse base.
This implements pbrt-v4's coateddiffuse material using random walk
sampling between the layers (LayeredBxDF algorithm).

# Fields
- `reflectance`: Diffuse reflectance of the base layer (RGB color)
- `u_roughness`: Roughness in U direction for the dielectric coating
- `v_roughness`: Roughness in V direction for the dielectric coating
- `thickness`: Thickness of the coating layer (affects absorption)
- `eta`: Index of refraction of the dielectric coating
- `albedo`: Single-scattering albedo of medium between layers (0 = no absorption)
- `g`: Henyey-Greenstein asymmetry parameter for medium scattering
- `max_depth`: Maximum random walk depth
- `n_samples`: Number of samples for estimating the BSDF
- `remap_roughness`: Whether to remap roughness to microfacet alpha
"""
# Non-parametric texture parameters: constant-vs-texture must not change the
# material's TYPE, or the per-material chit path compiles a separate shader per
# combination.
struct CoatedDiffuse{ReflT, URoughT, VRoughT, ThickT, AlbedoT, GT, DispT} <: Material
    reflectance::ReflT
    u_roughness::URoughT
    v_roughness::VRoughT
    thickness::ThickT
    eta::Float32            # Index of refraction
    albedo::AlbedoT
    g::GT
    max_depth::Int32
    n_samples::Int32
    remap_roughness::Bool
    displacement::DispT       # pbrt-v4 `Material::displacement` height field (NONE = flat)
end

# Positional constructor taking Ints for the two counters; the struct's own
# synthesized constructor converts every other argument through `TexHandle`.
function CoatedDiffuse(
    reflectance, u_roughness, v_roughness, thickness, eta::Float32,
    albedo, g, max_depth::Int, n_samples::Int, remap_roughness::Bool,
    displacement = TexHandle(),
)
    CoatedDiffuse(
        matparam(reflectance), matparam(u_roughness), matparam(v_roughness),
        matparam(thickness), eta, matparam(albedo), matparam(g),
        Int32(max_depth), Int32(n_samples), remap_roughness, matparam(displacement),
    )
end

"""
    CoatedDiffuse(; reflectance, roughness=0, thickness=0.01, eta=1.5, ...)

Create a coated diffuse material with keyword arguments.

# Arguments
- `reflectance`: Diffuse color (RGBSpectrum, tuple, or Texture)
- `roughness`: Surface roughness (scalar or (u,v) tuple)
- `thickness`: Coating thickness (default 0.01)
- `eta`: Index of refraction (default 1.5 for typical dielectric)
- `albedo`: Medium albedo for absorption (default 0 = no medium)
- `g`: HG asymmetry parameter (default 0 = isotropic)
- `max_depth`: Max random walk depth (default 10)
- `n_samples`: Number of samples (default 1)
- `remap_roughness`: Remap roughness to alpha (default true)

# Examples
```julia
# Simple coated diffuse (glossy plastic-like)
CoatedDiffuse(reflectance=(0.4, 0.45, 0.35), roughness=0)

# Rough coating
CoatedDiffuse(reflectance=(0.8, 0.2, 0.2), roughness=0.3)

# With absorbing medium
CoatedDiffuse(reflectance=(0.9, 0.9, 0.9), albedo=(0.8, 0.4, 0.2))
```
"""
function CoatedDiffuse(;
    reflectance = RGBSpectrum(0.5f0),
    roughness = 0f0,
    thickness = 0.01f0,
    eta::Real = 1.5f0,
    albedo = RGBSpectrum(0f0),
    g = 0f0,
    max_depth::Int = 10,
    n_samples::Int = 1,
    remap_roughness::Bool = true,
    bump = nothing,
)
    # Handle roughness - can be scalar, (u,v) tuple, or Texture{Float32}
    u_rough, v_rough = if roughness isa Tuple
        roughness[1], roughness[2]
    else
        roughness, roughness
    end

    CoatedDiffuse(
        matparam(reflectance),
        matparam(u_rough),
        matparam(v_rough),
        matparam(thickness),
        Float32(eta),
        matparam(albedo),
        matparam(g),
        max_depth,
        n_samples,
        remap_roughness,
        matparam(bump),
    )
end

# Mark as non-emissive
is_emissive(::CoatedDiffuse) = false


# ============================================================================
# Spectral BSDF Evaluation — CoatedDiffuse
# ============================================================================

"""
    sample_bsdf_spectral(mat::CoatedDiffuse, ...) -> SpectralBSDFSample

Sample CoatedDiffuse BSDF using pbrt-v4's LayeredBxDF random walk algorithm.
Top layer = dielectric interface, bottom layer = Lambertian diffuse.

Algorithm:
1. Sample entrance interface (dielectric)
2. If reflection at entrance: return immediately with pdfIsProportional=true
3. If transmission: start random walk through layers
4. At each depth: possibly scatter in medium, then sample interface
5. When ray exits through transmission: return the accumulated sample
6. Russian roulette for path termination

When `regularize=true`, the coating's microfacet alpha is increased to reduce fireflies.
"""
# CoatedDiffuseEvaluated — pbrt-v4 CoatedDiffuseBxDF analogue.
#
# This is the material that most needs it: the layered BSDF is stochastic, so
# `sample`/`evaluate` run a multi-bounce random walk, and every one of these
# six textures was re-read on each entry. `evaluate_bsdf_spectral` is called
# once per light sample in direct lighting on top of that.
struct CoatedDiffuseEvaluated
    reflectance::SpectralRadiance   # uplifted
    albedo::SpectralRadiance        # uplifted medium albedo
    thickness::Float32
    eta::Float32
    g::Float32
    alpha_x::Float32
    alpha_y::Float32
    has_medium::Bool
    max_depth::Int32
    n_samples::Int32
end

@propagate_inbounds function get_bxdf(
    mat::CoatedDiffuse, table::RGBToSpectrumTable, textures,
    tfc::TextureFilterContext, lambda::Wavelengths, regularize::Bool,
)
    refl_rgb   = eval_handle_spectrum(textures, mat.reflectance, tfc)
    albedo_rgb = eval_handle_spectrum(textures, mat.albedo, tfc)
    thickness  = max(eval_handle(textures, mat.thickness, tfc), eps(Float32))
    g_val      = clamp(eval_handle(textures, mat.g, tfc), -0.99f0, 0.99f0)

    u_roughness = eval_handle(textures, mat.u_roughness, tfc)
    v_roughness = eval_handle(textures, mat.v_roughness, tfc)
    alpha_x = mat.remap_roughness ? roughness_to_α(u_roughness) : u_roughness
    alpha_y = mat.remap_roughness ? roughness_to_α(v_roughness) : v_roughness
    if regularize
        alpha_x = regularize_alpha(alpha_x)
        alpha_y = regularize_alpha(alpha_y)
    end

    return CoatedDiffuseEvaluated(
        uplift_rgb(table, refl_rgb, lambda),
        uplift_rgb(table, albedo_rgb, lambda),
        thickness, mat.eta, g_val, alpha_x, alpha_y,
        !is_black(albedo_rgb), mat.max_depth, mat.n_samples)
end

@propagate_inbounds function sample_bsdf_spectral(
    bxdf::CoatedDiffuseEvaluated, ::RGBToSpectrumTable, textures,
    wo::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext,
    ::Wavelengths, sample_u::Point2f, rng_in::Float32,
    ::Bool = false,
)
    # Check for grazing angle
    wo_dot_n = dot(wo, n)
    if abs(wo_dot_n) < 1f-6
        return SpectralBSDFSample()
    end

    # All of it resolved once per hit in `get_bxdf`.
    eta             = bxdf.eta
    thickness       = bxdf.thickness
    g_val           = bxdf.g
    alpha_x         = bxdf.alpha_x
    alpha_y         = bxdf.alpha_y
    refl_spectral   = bxdf.reflectance
    albedo_spectral = bxdf.albedo
    has_medium      = bxdf.has_medium
    max_depth       = Int(bxdf.max_depth)

    # Build coordinate system from shading normal
    tangent, bitangent = shading_frame(n, dpdus)

    # Transform wo to local space
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), wo_dot_n)

    # Two-sided handling: flip if entering from below
    flip_wi = wo_local[3] < 0f0
    if flip_wi
        wo_local = -wo_local
    end

    # Determine entrance interface (top for wo.z > 0, which is always true after flip)
    entered_top = true  # After flip, wo_local.z > 0, so we enter from top

    # === Sample entrance interface ===
    # Use rng_in and sample_u for entrance sampling
    bs = sample_dielectric_interface(wo_local, rng_in, sample_u, alpha_x, alpha_y, eta, BXDF_ALL)

    if !bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0
        return SpectralBSDFSample()
    end

    # If reflection at entrance: return immediately (pdfIsProportional case)
    if bs.is_reflection
        wi_local = bs.wi
        if flip_wi
            wi_local = -wi_local
        end
        wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
        wi = normalize(wi)

        # pdfIsProportional=true — LayeredBxDF returns proportional PDF
        flags = bs.is_specular ? BXDF_SPECULAR_REFLECTION : BXDF_GLOSSY_REFLECTION
        return SpectralBSDFSample(bs.f, wi, bs.pdf, flags, 1f0, true, false)
    end

    # === Begin random walk through layers ===
    w = bs.wi
    specular_path = bs.is_specular
    f = bs.f * abs(w[3])  # f * AbsCosTheta(wi)
    pdf = bs.pdf
    z = entered_top ? thickness : 0f0  # Start at bottom of coating after transmission

    # Initialize deterministic RNG for random walk (GPU-compatible functional style)
    # Use Hash(seed, wo) combined with Hash(uc, u) for reproducibility
    seed = UInt64(0)  # pbrt uses GetOptions().seed, we use 0
    rng = pcg32_init(pbrt_hash(seed, wo_local), pbrt_hash(rng_in, sample_u))

    for depth in 0:(max_depth-1)
        # Possibly terminate with Russian Roulette
        rr_beta = max_component(f) / pdf
        if depth > 3 && rr_beta < 0.25f0
            q = max(0f0, 1f0 - rr_beta)
            rr_val, rng = pcg32_uniform_f32(rng)
            if rr_val < q
                return SpectralBSDFSample()
            end
            pdf *= 1f0 - q
        end

        if w[3] == 0f0
            return SpectralBSDFSample()
        end

        if has_medium
            # Sample potential scattering in medium
            sigma_t = 1f0
            exp_u, rng = pcg32_uniform_f32(rng)
            dz = sample_exponential(exp_u, sigma_t / abs(w[3]))
            zp = w[3] > 0f0 ? (z + dz) : (z - dz)

            if zp == z
                return SpectralBSDFSample()
            end

            if 0f0 < zp && zp < thickness
                # Scattering event within the medium
                phase_u1, rng = pcg32_uniform_f32(rng)
                phase_u2, rng = pcg32_uniform_f32(rng)
                wi_phase, phase_p = sample_hg_phase_spectral(g_val, -w, Point2f(phase_u1, phase_u2))
                if phase_p == 0f0 || wi_phase[3] == 0f0
                    return SpectralBSDFSample()
                end
                f = f * albedo_spectral * phase_p
                pdf *= phase_p
                specular_path = false
                w = wi_phase
                z = zp
                continue
            end

            # Clamp to layer boundary
            z = clamp(zp, 0f0, thickness)
        else
            # No medium: advance directly to other interface
            z = (z == thickness) ? 0f0 : thickness
            f = f * layer_transmittance(thickness, w)
        end

        # Determine which interface we're at
        at_bottom = z == 0f0

        # Sample interface BSDF
        uc, rng = pcg32_uniform_f32(rng)
        u1, rng = pcg32_uniform_f32(rng)
        u2, rng = pcg32_uniform_f32(rng)
        u = Point2f(u1, u2)

        if at_bottom
            # Sample diffuse base (reflection only)
            bs_interface = sample_diffuse_interface(-w, u, refl_spectral, BXDF_ALL)
        else
            # Sample dielectric top (can reflect or transmit)
            bs_interface = sample_dielectric_interface(-w, uc, u, alpha_x, alpha_y, eta, BXDF_ALL)
        end

        if !bs_interface.valid || bs_interface.pdf == 0f0 || bs_interface.wi[3] == 0f0
            return SpectralBSDFSample()
        end

        f = f * bs_interface.f
        pdf *= bs_interface.pdf
        specular_path = specular_path && bs_interface.is_specular
        w = bs_interface.wi

        # Check if path has exited the layers (transmission through an interface)
        if !bs_interface.is_reflection
            # Ray has exited - determine final direction flags
            wi_local = w
            if flip_wi
                wi_local = -wi_local
            end

            wi = tangent * wi_local[1] + bitangent * wi_local[2] + n * wi_local[3]
            wi = normalize(wi)

            # pdfIsProportional=true for LayeredBxDF
            is_refl = same_hemisphere(wo_local, flip_wi ? -w : w)
            flags = if specular_path
                is_refl ? BXDF_SPECULAR_REFLECTION : BXDF_SPECULAR_TRANSMISSION
            else
                is_refl ? BXDF_GLOSSY_REFLECTION : BXDF_GLOSSY_TRANSMISSION
            end
            # pbrt-v4 hardcodes eta=1 for LayeredBxDF exit (bxdfs.h:768)
            return SpectralBSDFSample(f, wi, pdf, flags, 1f0, true, false)
        end

        # Continuing random walk: multiply by AbsCosTheta for next segment
        f = f * abs(bs_interface.wi[3])
    end

    # Max depth reached without exiting
    return SpectralBSDFSample()
end

"""
    evaluate_bsdf_spectral(table, mat::CoatedDiffuse, textures, wo, wi, n, uv, lambda) -> (f, pdf)

Evaluate CoatedDiffuse BSDF using pbrt-v4's LayeredBxDF::f random walk algorithm.
Exact port of pbrt-v4 bxdfs.h lines 477-652.
"""
@propagate_inbounds function evaluate_bsdf_spectral(
    bxdf::CoatedDiffuseEvaluated, ::RGBToSpectrumTable, textures,
    wo::Vec3f, wi::Vec3f, n::Vec3f, dpdus::Vec3f, ::TextureFilterContext, ::Wavelengths,
    ::Bool = false,
)
    eta             = bxdf.eta
    thickness       = bxdf.thickness
    g_val           = bxdf.g
    alpha_x         = bxdf.alpha_x
    alpha_y         = bxdf.alpha_y
    refl_spectral   = bxdf.reflectance
    albedo_spectral = bxdf.albedo
    has_medium      = bxdf.has_medium
    n_samples       = Int(bxdf.n_samples)
    max_depth       = Int(bxdf.max_depth)

    tangent, bitangent = shading_frame(n, dpdus)
    wo_local = Vec3f(dot(wo, tangent), dot(wo, bitangent), dot(wo, n))
    wi_local = Vec3f(dot(wi, tangent), dot(wi, bitangent), dot(wi, n))

    # twoSided: flip if entering from below
    if wo_local[3] < 0f0
        wo_local = -wo_local
        wi_local = -wi_local
    end
    if abs(wo_local[3]) < 1f-6 || abs(wi_local[3]) < 1f-6
        return (SpectralRadiance(), 0f0)
    end

    entered_top = true
    same_hemi = same_hemisphere(wo_local, wi_local)
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    # exitInterface / nonExitInterface determination (pbrt lines 494-503)
    # For CoatedDiffuse: top=dielectric, bottom=diffuse
    # exit_at_bottom = SameHemisphere(wo,wi) XOR enteredTop
    exit_at_bottom = same_hemi ⊻ entered_top
    exit_z = exit_at_bottom ? 0f0 : thickness
    # exitInterface is bottom (diffuse) when exit_at_bottom, top (dielectric) otherwise
    # nonExitInterface is the other one
    exit_is_specular = exit_at_bottom ? false : is_smooth   # diffuse is never specular
    nonexit_is_specular = exit_at_bottom ? is_smooth : false

    # pbrt line 505-507: Account for reflection at entrance interface
    f_result = SpectralRadiance()
    if same_hemi
        enter_f, _ = eval_dielectric_interface(wo_local, wi_local, alpha_x, alpha_y, eta)
        f_result = f_result + enter_f * Float32(n_samples)
    end

    # pbrt line 509-512: RNG
    rng = pcg32_init(pbrt_hash(UInt64(0), wo_local), pbrt_hash(wi_local))

    for s in 1:n_samples
        # pbrt line 517-522: Sample transmission through entrance (top) interface
        uc, rng = pcg32_uniform_f32(rng)
        u1, rng = pcg32_uniform_f32(rng)
        u2, rng = pcg32_uniform_f32(rng)
        wos = sample_dielectric_interface(wo_local, uc, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
        if !wos.valid || wos.pdf == 0f0 || wos.wi[3] == 0f0
            continue
        end

        # pbrt line 524-529: Sample virtual light from exit interface
        uc, rng = pcg32_uniform_f32(rng)
        u1, rng = pcg32_uniform_f32(rng)
        u2, rng = pcg32_uniform_f32(rng)
        # pbrt-v4: wis = exitInterface.Sample_f(wi, ..., !mode, Transmission)
        # !mode = Importance (no 1/etap² correction)
        wis = if exit_at_bottom
            sample_diffuse_interface(wi_local, Point2f(u1, u2), refl_spectral, BXDF_TRANSMISSION)
        else
            sample_dielectric_interface(wi_local, uc, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_TRANSMISSION, false)
        end
        if !wis.valid || wis.pdf == 0f0 || wis.wi[3] == 0f0
            continue
        end

        # pbrt line 531-535: Initialize random walk state
        beta = wos.f * abs(wos.wi[3]) / wos.pdf
        z = entered_top ? thickness : 0f0
        w = wos.wi

        for depth in 0:(max_depth-1)
            # pbrt line 542-550: Russian Roulette
            if depth > 3 && max_component(beta) < 0.25f0
                q = max(0f0, 1f0 - max_component(beta))
                rr_val, rng = pcg32_uniform_f32(rng)
                if rr_val < q
                    break
                end
                beta = beta / (1f0 - q)
            end

            # pbrt line 552-600: Account for media between layers
            if !has_medium
                # pbrt line 553-556: No medium — advance to next boundary
                z = (z == thickness) ? 0f0 : thickness
                beta = beta * layer_transmittance(thickness, w)
            else
                # pbrt line 558-599: Sample medium scattering
                sigma_t = 1f0
                exp_u, rng = pcg32_uniform_f32(rng)
                dz = sample_exponential(exp_u, sigma_t / abs(w[3]))
                zp = w[3] > 0f0 ? (z + dz) : (z - dz)
                if zp == z
                    continue
                end
                if 0f0 < zp && zp < thickness
                    # pbrt line 567-573: NEE contribution through exit using wis
                    wt = 1f0
                    if !exit_is_specular
                        wt = power_heuristic(1, wis.pdf, 1, hg_phase_pdf(g_val, dot(-w, -wis.wi)))
                    end
                    phase_val = hg_phase_pdf(g_val, dot(-w, -wis.wi))
                    f_result = f_result + beta * albedo_spectral * phase_val * wt *
                               layer_transmittance(zp - exit_z, wis.wi) * wis.f / wis.pdf

                    # pbrt line 575-582: Sample phase function
                    phase_u1, rng = pcg32_uniform_f32(rng)
                    phase_u2, rng = pcg32_uniform_f32(rng)
                    wi_phase, phase_p = sample_hg_phase_spectral(g_val, -w, Point2f(phase_u1, phase_u2))
                    if phase_p == 0f0 || wi_phase[3] == 0f0
                        continue
                    end
                    beta = beta * albedo_spectral * phase_p / phase_p
                    w = wi_phase
                    z = zp

                    # pbrt line 584-595: NEE through exit after phase scattering
                    if ((z < exit_z && w[3] > 0f0) || (z > exit_z && w[3] < 0f0)) && !exit_is_specular
                        f_exit, _ = if exit_at_bottom
                            eval_diffuse_interface(-w, wi_local, refl_spectral)
                        else
                            eval_dielectric_interface(-w, wi_local, alpha_x, alpha_y, eta)
                        end
                        if max_component(f_exit) > 0f0
                            exit_pdf = if exit_at_bottom
                                pdf_diffuse_interface(-w, wi_local)
                            else
                                pdf_dielectric_interface(-w, wi_local, alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
                            end
                            wt2 = power_heuristic(1, phase_p, 1, exit_pdf)
                            f_result = f_result + beta * layer_transmittance(zp - exit_z, wi_phase) * f_exit * wt2
                        end
                    end
                    continue
                end
                z = clamp(zp, 0f0, thickness)
            end

            # pbrt line 602-648: Account for scattering at appropriate interface
            if z == exit_z
                # pbrt line 603-611: At exit interface — sample reflection to continue walk
                uc, rng = pcg32_uniform_f32(rng)
                u1, rng = pcg32_uniform_f32(rng)
                u2, rng = pcg32_uniform_f32(rng)
                bs = if exit_at_bottom
                    sample_diffuse_interface(-w, Point2f(u1, u2), refl_spectral, BXDF_REFLECTION)
                else
                    sample_dielectric_interface(-w, uc, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_REFLECTION)
                end
                if !bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0
                    break
                end
                beta = beta * bs.f * abs(bs.wi[3]) / bs.pdf
                w = bs.wi
            else
                # pbrt line 613-648: At non-exit interface — NEE + sample new direction

                # pbrt line 615-623: NEE along presampled wis direction
                if !nonexit_is_specular
                    f_nee, _ = if z == thickness
                        eval_dielectric_interface(-w, -wis.wi, alpha_x, alpha_y, eta)
                    else
                        eval_diffuse_interface(-w, -wis.wi, refl_spectral)
                    end
                    if max_component(f_nee) > 0f0
                        wt = 1f0
                        if !exit_is_specular
                            nee_pdf = if z == thickness
                                pdf_dielectric_interface(-w, -wis.wi, alpha_x, alpha_y, eta)
                            else
                                pdf_diffuse_interface(-w, -wis.wi)
                            end
                            wt = power_heuristic(1, wis.pdf, 1, nee_pdf)
                        end
                        f_result = f_result + beta * f_nee * abs(wis.wi[3]) * wt *
                                   layer_transmittance(thickness, wis.wi) * wis.f / wis.pdf
                    end
                end

                # pbrt line 625-633: Sample new direction at nonExitInterface
                uc, rng = pcg32_uniform_f32(rng)
                u1, rng = pcg32_uniform_f32(rng)
                u2, rng = pcg32_uniform_f32(rng)
                bs = if z == thickness
                    sample_dielectric_interface(-w, uc, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_REFLECTION)
                else
                    sample_diffuse_interface(-w, Point2f(u1, u2), refl_spectral, BXDF_REFLECTION)
                end
                if !bs.valid || bs.pdf == 0f0 || bs.wi[3] == 0f0
                    break
                end
                beta = beta * bs.f * abs(bs.wi[3]) / bs.pdf
                w = bs.wi

                # pbrt line 635-647: NEE through exit after scattering
                if !exit_is_specular
                    f_exit, _ = if exit_at_bottom
                        eval_diffuse_interface(-w, wi_local, refl_spectral)
                    else
                        eval_dielectric_interface(-w, wi_local, alpha_x, alpha_y, eta)
                    end
                    if max_component(f_exit) > 0f0
                        wt3 = 1f0
                        if !nonexit_is_specular
                            exit_pdf3 = if exit_at_bottom
                                pdf_diffuse_interface(-w, wi_local)
                            else
                                pdf_dielectric_interface(-w, wi_local, alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
                            end
                            wt3 = power_heuristic(1, bs.pdf, 1, exit_pdf3)
                        end
                        f_result = f_result + beta * layer_transmittance(thickness, bs.wi) * f_exit * wt3
                    end
                end
            end
        end
    end

    # pbrt line 652
    f_result = f_result / Float32(n_samples)

    pdf = pdf_layered_bsdf(wo_local, wi_local, alpha_x, alpha_y, eta, n_samples, max_depth, refl_spectral, has_medium, g_val, thickness)
    return (f_result, pdf)
end

"""
    pdf_layered_bsdf(...) -> Float32

Compute PDF for LayeredBxDF using Monte Carlo estimation.
This is a simplified version of pbrt-v4's LayeredBxDF::PDF.
"""
@propagate_inbounds function pdf_layered_bsdf(
    wo::Vec3f, wi::Vec3f,
    alpha_x::Float32, alpha_y::Float32, eta::Float32,
    n_samples::Int, max_depth::Int,
    refl_spectral::SpectralRadiance,
    has_medium::Bool, g_val::Float32, thickness::Float32
)
    # Initialize RNG (GPU-compatible functional style)
    seed = UInt64(0)
    rng = pcg32_init(pbrt_hash(seed, wi), pbrt_hash(wo))

    entered_top = true
    same_hemi = same_hemisphere(wo, wi)
    is_smooth = trowbridge_reitz_effectively_smooth(alpha_x, alpha_y)

    pdf_sum = 0f0

    # Add entrance reflection PDF (same hemisphere)
    if same_hemi
        if is_smooth
            # Specular - delta PDF contribution
            pdf_sum += Float32(n_samples) * 0f0  # Delta has no continuous PDF
        else
            pdf_sum += Float32(n_samples) * pdf_dielectric_interface(wo, wi, alpha_x, alpha_y, eta, BXDF_REFLECTION)
        end
    end

    for s in 1:n_samples
        if same_hemi
            # TRT term
            # Sample transmission through top
            uc1, rng = pcg32_uniform_f32(rng)
            u1, rng = pcg32_uniform_f32(rng)
            u2, rng = pcg32_uniform_f32(rng)
            wos = sample_dielectric_interface(wo, uc1, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)

            uc2, rng = pcg32_uniform_f32(rng)
            u3, rng = pcg32_uniform_f32(rng)
            u4, rng = pcg32_uniform_f32(rng)
            wis = sample_dielectric_interface(wi, uc2, Point2f(u3, u4), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)

            if wos.valid && wos.pdf > 0f0 && wis.valid && wis.pdf > 0f0
                if is_smooth
                    # Specular top - just use bottom PDF
                    pdf_sum += pdf_diffuse_interface(-wos.wi, -wis.wi)
                else
                    # MIS between paths
                    u5, rng = pcg32_uniform_f32(rng)
                    u6, rng = pcg32_uniform_f32(rng)
                    rs = sample_diffuse_interface(-wos.wi, Point2f(u5, u6), refl_spectral, BXDF_ALL)
                    if rs.valid && rs.pdf > 0f0
                        r_pdf = pdf_diffuse_interface(-wos.wi, -wis.wi)
                        wt = power_heuristic(1, wis.pdf, 1, r_pdf)
                        pdf_sum += wt * r_pdf

                        t_pdf = pdf_dielectric_interface(-rs.wi, wi, alpha_x, alpha_y, eta)
                        wt2 = power_heuristic(1, rs.pdf, 1, t_pdf)
                        pdf_sum += wt2 * t_pdf
                    end
                end
            end
        else
            # TT term
            uc1, rng = pcg32_uniform_f32(rng)
            u1, rng = pcg32_uniform_f32(rng)
            u2, rng = pcg32_uniform_f32(rng)
            wos = sample_dielectric_interface(wo, uc1, Point2f(u1, u2), alpha_x, alpha_y, eta, BXDF_TRANSMISSION)
            if !wos.valid || wos.pdf == 0f0 || wos.is_reflection
                continue
            end

            u3, rng = pcg32_uniform_f32(rng)
            u4, rng = pcg32_uniform_f32(rng)
            wis = sample_diffuse_interface(wi, Point2f(u3, u4), refl_spectral, BXDF_TRANSMISSION)
            if !wis.valid || wis.pdf == 0f0 || wis.is_reflection
                continue
            end

            if is_smooth
                pdf_sum += pdf_diffuse_interface(-wos.wi, wi)
            else
                pdf_sum += (pdf_dielectric_interface(wo, -wis.wi, alpha_x, alpha_y, eta) +
                           pdf_diffuse_interface(-wos.wi, wi)) / 2f0
            end
        end
    end

    # pbrt-v4: Lerp(0.9, 1/(4π), pdfSum/nSamples) = 0.1/(4π) + 0.9 * pdfSum/nSamples
    # Hikari lerp(v1, v2, t) = (1-t)*v1 + t*v2, so we need lerp(1/(4π), pdf_sum/nSamples, 0.9)
    return lerp(1f0 / (4f0 * Float32(π)), pdf_sum / Float32(n_samples), 0.9f0)
end

# ============================================================================
# Plastic — convenience preset for CoatedDiffuse (like Gold() for Conductor)
# ============================================================================

"""
    Plastic(; color=(0.5, 0.5, 0.5), roughness=0.1, eta=1.5)

Create a plastic-like material. Returns a `CoatedDiffuse` with a dielectric
coating (eta ≈ 1.5) over a diffuse base — the standard way to represent
plastic in pbrt-v4.

# Examples
```julia
Plastic()                                    # grey plastic
Plastic(color=(0.8, 0.1, 0.1))              # red plastic
Plastic(color=(0.9, 0.9, 0.9), roughness=0.01)  # glossy white
```
"""
function Plastic(; color=(0.5f0, 0.5f0, 0.5f0), roughness=0.1f0, eta=1.5f0)
    CoatedDiffuse(
        reflectance=color,
        roughness=Float32(roughness),
        eta=Float32(eta),
    )
end

# ============================================================================
# Surface Alpha Evaluation
# ============================================================================

"""
    get_surface_alpha(mat::CoatedDiffuse, textures, uv) -> Float32

The cutout alpha of the coat's base texture, so alpha-masked geometry is cut the
same way it is under [`Diffuse`](@ref).

Without this the generic `Material` fallback returns 1 and the surface is fully
opaque. That is a silent loss on any asset whose silhouette lives in its alpha
channel: the raven demo model paints its feather shapes there — 38% of its map is
fully transparent and 30% of its triangles are partly cut — so giving the bird a
coat turned every feather card back into a solid quad.
"""
@propagate_inbounds function get_surface_alpha(mat::CoatedDiffuse, textures, uv::Point2f)
    refl = eval_handle_spectrum(textures, mat.reflectance, TextureFilterContext(uv))
    return get_alpha(refl)
end
