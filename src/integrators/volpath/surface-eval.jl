# Surface material evaluation for VolPath
# Handles BSDF sampling, direct lighting, and path continuation at surface hits
#
# NOTE: `materials` is a StaticMultiTypeSet containing both materials and textures.
# Use eval_tex(materials, field, uv) to sample textures via TextureRef.

# ============================================================================
# Texture Filtering Derivatives (pbrt-v4 style)
# ============================================================================

"""
    approximate_dp_dxy(pi, n, camera, samples_per_pixel) -> (dpdx, dpdy)

Approximate screen-space position derivatives at intersection point.
Following pbrt-v4's Camera::Approximate_dp_dxy method.

This estimates how much the surface position changes per pixel, which is used
for texture filtering (mipmap level selection). The approximation assumes the
surface is locally planar near the intersection point.

For a perspective camera, this is approximately:
    dp/dscreen ≈ distance * tan(fov/2) / (resolution/2)

Arguments:
- `pi`: Intersection point in world space
- `n`: Surface normal at intersection
- `camera`: Camera with dx_camera, dy_camera precomputed
- `samples_per_pixel`: Number of samples per pixel (for scaling)

Returns (dpdx, dpdy) - approximate change in position per screen pixel.
"""
@propagate_inbounds function approximate_dp_dxy(
    pi::Point3f, n::Vec3f, camera, samples_per_pixel::Int32
)
    # Get camera_to_world transform (generic accessor handles different camera types)
    camera_to_world = get_camera_to_world(camera)

    # For perspective camera, approximate dpdx/dpdy based on distance and fov
    # The camera has precomputed dx_camera and dy_camera (change per pixel in camera space)

    # Get camera position (world space origin)
    camera_pos = Raycore.transform_point(camera_to_world.m, Point3f(0f0))

    # Distance from camera to intersection point
    to_point = Vec3f(pi - camera_pos)
    dist = sqrt(dot(to_point, to_point))

    # Hikari's `dx_camera` is the camera-space per-raster-pixel displacement on
    # the plane `raster_to_camera` maps the film onto, not the angular size of a
    # pixel. To project it to world displacement at the actual hit distance we
    # divide by that plane's depth; otherwise the world displacement comes out
    # ~100× too small and the per-pixel UV footprint falls below
    # BUMP_DEFAULT_DELTA — so the bump fallback (5e-4) kicks in instead of the
    # real screen-space derivative, leaving the gold-dome conductor with
    # coherent mirror highlights.
    #
    # That plane sits at `2 * near`, not `near`, so dividing by the camera's
    # near distance made every footprint in the renderer exactly 2x too wide —
    # verified against differentials reconstructed from real neighbouring-pixel
    # rays, which `surface_dp_dxy` now matches to 0.05%.
    #
    # pbrt-v4 scales BOTH differential paths by max(.125, 1/sqrt(spp)):
    # true camera ray diffs via `ray.ScaleDifferentials(rayDiffScale)`
    # (cpu/integrators.cpp:251) and the fallback via `sppScale` inside
    # `Approximate_dp_dxy` (cameras.h:175). An earlier comment here claimed
    # the exact-diff path is unscaled and skipped sppScale — that left the
    # footprint 8× too large at 256 spp and over-smoothed every bump map
    # (shadow_bumpgold_dome_over_velvet pins this). The sub-texel-du
    # concern that motivated skipping it only applied to the old
    # nearest-neighbour `sample_texture_data`; bilinear sampling returns
    # the correct local slope for sub-texel finite differences.
    spp_scale = max(0.125f0, 1f0 / sqrt(Float32(samples_per_pixel)))
    scale = spp_scale * dist / abs(raster_plane_z(camera))

    # Transform dx_camera and dy_camera to world space
    # These represent how the ray direction changes per pixel
    dx_world = Raycore.transform_direction(camera_to_world.m, camera.dx_camera)
    dy_world = Raycore.transform_direction(camera_to_world.m, camera.dy_camera)

    # Project onto the tangent plane at the intersection
    # dpdx ≈ scale * (dx_world - n * dot(n, dx_world))
    dpdx = scale * (dx_world - n * dot(n, dx_world))
    dpdy = scale * (dy_world - n * dot(n, dy_world))

    return dpdx, dpdy
end

"""
    surface_dp_dxy(camera, ray_o, ray_d, p, n, samples_per_pixel, depth) -> (dpdx, dpdy)

Screen-space position derivatives at a hit point.

pbrt-v4 `SurfaceInteraction::ComputeDifferentials` (interaction.cpp:48) uses the
ray's TRUE differentials whenever it has them, and only falls back to
`Camera::Approximate_dp_dxy` when it does not. Camera rays always have them
(`GenerateRayDifferential`); indirect bounces never do, because the path
integrators do not propagate differentials. So `depth == 0` gets the exact
footprint and everything deeper gets the approximation — matching pbrt on both
branches rather than approximating everywhere.

This matters well beyond mipmap selection: the footprint sets
`δu = 0.5(|dudx|+|dudy|)`, the finite-difference step `perturb_bump_frame` uses
for `∂h/∂u`. A wrong footprint tilts the bumped normal, which MOVES specular
highlights rather than just dimming them. Measured on crown.pbrt as a coherent
dipole on the displaced dome panels.

Only `PerspectiveCamera` implements the exact form — it is the camera the pbrt
importer builds, and the reconstruction below depends on its near-plane
convention. Other cameras keep the approximation.
"""
@propagate_inbounds surface_dp_dxy(camera, ray_o::Point3f, ray_d::Vec3f,
                                   p::Point3f, n::Vec3f,
                                   samples_per_pixel::Int32, depth::Int32) =
    approximate_dp_dxy(p, n, camera, samples_per_pixel)

@propagate_inbounds function surface_dp_dxy(camera::PerspectiveCamera,
                                            ray_o::Point3f, ray_d::Vec3f,
                                            p::Point3f, n::Vec3f,
                                            samples_per_pixel::Int32, depth::Int32)
    depth == Int32(0) || return approximate_dp_dxy(p, n, camera, samples_per_pixel)

    ctw = get_camera_to_world(camera)
    d_cam = normalize(Raycore.transform_direction(ctw.inv_m, ray_d))
    # Degenerate: ray parallel to the image plane. Nothing to reconstruct.
    abs(d_cam[3]) < 1f-9 && return approximate_dp_dxy(p, n, camera, samples_per_pixel)

    # Hikari's camera looks down -z (`generate_ray`: t = -focal_distance / d[3]),
    # pbrt looks down +z, so the near and focal planes sit at NEGATIVE z here.
    #
    # `dx_camera`/`dy_camera` are offsets on whatever plane `raster_to_camera`
    # maps z=0 onto, and `p_camera` has to land on that SAME plane or the offsets
    # are the wrong size. That plane is at `2 * near`, not `near`, so derive it
    # rather than assume it.
    z_ref = raster_plane_z(camera)
    focal = camera.core.focal_distance
    uses_lens = camera.core.lens_radius > 0f0

    # Recover `p_camera`, the near-plane point this pixel's ray passed through —
    # `dx_camera`/`dy_camera` are offsets on that same plane, so the reconstruction
    # has to land on it.
    o_cam = Raycore.transform_point(ctw.inv_m, ray_o)
    p_camera = if uses_lens
        # With a lens the ray runs lens_point -> focal point, so the unlensed
        # direction is the one from the camera origin through the focal point.
        # `p_focus` is proportional to `p_camera` (cameras.cpp:441-443), so
        # rescaling it to the near plane recovers `p_camera` exactly.
        t_focus = (-focal - o_cam[3]) / d_cam[3]
        p_focus = o_cam + d_cam * t_focus
        abs(p_focus[3]) < 1f-9 && return approximate_dp_dxy(p, n, camera, samples_per_pixel)
        p_focus * (z_ref / p_focus[3])
    else
        d_cam * (z_ref / d_cam[3])
    end

    # pbrt-v4 cameras.cpp:451-474 — offset rays through the neighbouring pixels.
    # Both differentials share the primary ray's origin in either branch.
    px_cam = Vec3f(p_camera) + camera.dx_camera
    py_cam = Vec3f(p_camera) + camera.dy_camera
    rx_d_cam, ry_d_cam = if uses_lens
        dx = normalize(px_cam); dy = normalize(py_cam)
        (abs(dx[3]) < 1f-9 || abs(dy[3]) < 1f-9) &&
            return approximate_dp_dxy(p, n, camera, samples_per_pixel)
        (normalize(dx * (-focal / dx[3]) - Vec3f(o_cam)),
         normalize(dy * (-focal / dy[3]) - Vec3f(o_cam)))
    else
        (normalize(px_cam), normalize(py_cam))
    end

    # RayDifferential::ScaleDifferentials, applied by the integrator at
    # cpu/integrators.cpp:251. Origins coincide with the primary ray's, so only
    # the directions move.
    s = max(0.125f0, 1f0 / sqrt(Float32(samples_per_pixel)))
    rx_d = Raycore.transform_direction(ctw.m, d_cam + (rx_d_cam - d_cam) * s)
    ry_d = Raycore.transform_direction(ctw.m, d_cam + (ry_d_cam - d_cam) * s)

    # pbrt-v4 interaction.cpp:52-61 — intersect both offset rays with the tangent
    # plane at the hit and difference against it.
    n_rx = dot(n, rx_d); n_ry = dot(n, ry_d)
    (abs(n_rx) < 1f-9 || abs(n_ry) < 1f-9) &&
        return approximate_dp_dxy(p, n, camera, samples_per_pixel)
    num = dot(n, Vec3f(p)) - dot(n, Vec3f(ray_o))
    tx = num / n_rx
    ty = num / n_ry
    (isfinite(tx) && isfinite(ty)) ||
        return approximate_dp_dxy(p, n, camera, samples_per_pixel)
    return (Vec3f(ray_o) + rx_d * tx) - Vec3f(p), (Vec3f(ray_o) + ry_d * ty) - Vec3f(p)
end

"""
    bump_filter_context(camera, work, geom, samples_per_pixel, face_idx, bary) -> TextureFilterContext

The differentials + UV-derivative + context construction that every trace kernel
needs before perturbing the shading frame. Six call sites across the software and
hardware paths had this open-coded identically. `face_idx` and `bary` are the
hit's: a vertex-colour bump height indexes with them.
"""
@propagate_inbounds function bump_filter_context(camera, work, geom, samples_per_pixel::Int32,
                                                 face_idx::UInt32, bary::SVector{3, Float32})
    dpdx, dpdy = surface_dp_dxy(camera, work.ray.o, work.ray.d, geom.pi, geom.n,
                                samples_per_pixel, work.depth)
    dudx, dudy, dvdx, dvdy = compute_uv_derivatives(geom.dpdu, geom.dpdv, dpdx, dpdy)
    return TextureFilterContext(geom.uv, dudx, dudy, dvdx, dvdy, face_idx, bary)
end

"""
    compute_uv_derivatives(dpdu, dpdv, dpdx, dpdy) -> (dudx, dudy, dvdx, dvdy)

Compute UV derivatives from position derivatives using least-squares solve.
Following pbrt-v4's SurfaceInteraction::ComputeDifferentials.

Given:
- dpdu, dpdv: How position changes with UV (∂p/∂u, ∂p/∂v)
- dpdx, dpdy: How position changes with screen pixel (∂p/∂x, ∂p/∂y)

Solve for:
- dudx, dudy: How u changes with screen pixel (∂u/∂x, ∂u/∂y)
- dvdx, dvdy: How v changes with screen pixel (∂v/∂x, ∂v/∂y)

Uses the normal equations: (A^T A) [du/dx; dv/dx]^T = A^T [dpdx]
where A = [dpdu | dpdv] is a 3x2 matrix.
"""
@propagate_inbounds function compute_uv_derivatives(
    dpdu::Vec3f, dpdv::Vec3f, dpdx::Vec3f, dpdy::Vec3f
)
    # Compute A^T A (2x2 matrix)
    ata00 = dot(dpdu, dpdu)
    ata01 = dot(dpdu, dpdv)
    ata11 = dot(dpdv, dpdv)

    # Compute determinant and check for degeneracy
    det = ata00 * ata11 - ata01 * ata01
    if abs(det) < 1f-10
        return (0f0, 0f0, 0f0, 0f0)
    end
    inv_det = 1f0 / det

    # Compute A^T b for x direction
    atb0x = dot(dpdu, dpdx)
    atb1x = dot(dpdv, dpdx)

    # Compute A^T b for y direction
    atb0y = dot(dpdu, dpdy)
    atb1y = dot(dpdv, dpdy)

    # Solve using Cramer's rule (2x2 system)
    # [dudx]   [ata11  -ata01] [atb0x]
    # [dvdx] = [-ata01  ata00] [atb1x] * inv_det
    dudx = (ata11 * atb0x - ata01 * atb1x) * inv_det
    dvdx = (ata00 * atb1x - ata01 * atb0x) * inv_det
    dudy = (ata11 * atb0y - ata01 * atb1y) * inv_det
    dvdy = (ata00 * atb1y - ata01 * atb0y) * inv_det

    # Clamp to reasonable values (following pbrt-v4)
    clamp_val = 1f8
    dudx = clamp(dudx, -clamp_val, clamp_val)
    dvdx = clamp(dvdx, -clamp_val, clamp_val)
    dudy = clamp(dudy, -clamp_val, clamp_val)
    dvdy = clamp(dvdy, -clamp_val, clamp_val)

    return (dudx, dudy, dvdx, dvdy)
end

# TextureFilterContext is defined in textures/texture-ref.jl

"""
    compute_texture_filter_context(work, camera, samples_per_pixel) -> TextureFilterContext

Compute texture filtering context from material evaluation work item.
Uses approximate screen-space derivatives for proper mipmap selection.
"""
@propagate_inbounds function compute_texture_filter_context(
    work::VPMaterialEvalWorkItem, camera, samples_per_pixel::Int32
)
    # Compute screen-space position derivatives
    dpdx, dpdy = approximate_dp_dxy(work.pi, work.n, camera, samples_per_pixel)

    # Compute UV derivatives from position derivatives
    dudx, dudy, dvdx, dvdy = compute_uv_derivatives(work.dpdu, work.dpdv, dpdx, dpdy)

    return TextureFilterContext(work.uv, dudx, dudy, dvdx, dvdy, work.face_idx, work.bary)
end

# ============================================================================
# Surface Shading (pbrt-v4 wavefront/surfscatter.cpp:41 pattern)
# ============================================================================
#
# One kernel per concrete material type, monomorphised via `ForEachType`, so
# a kernel's SPIR-V contains exactly one material's BSDF. The alternative — a
# single kernel with every material's BSDF behind a `with_index` switch —
# measured 128 registers on RTX 4000 Ada, right at the occupancy cliff (above
# 128 the per-SM thread count drops from 512 to 256).
#
# This is the ONLY surface-shading implementation, and the closest-hit shaders
# call the same two inner functions. A second, `with_index`-based copy would be
# dispatched for exactly one configuration (hardware RT on a scene with media),
# which is how such a copy drifts: move every BSDF onto a `get_bxdf` carrier,
# update one copy, and the other keeps calling the raw material and silently
# matches a gray-Lambertian `::Material` catch-all — a
# specular medium boundary became a diffuse wall and `medium_smoke_point`
# rendered at 0.037 of reference on HW while SW was correct. Everything that
# copy did beyond shading — MixMaterial resolution, the null-material medium
# swap — already happens at the push site for both producers
# (`vp_trace_and_shade_kernel!` and delta tracking's `vp_sample_medium_kernel!`),
# so it was redundant as well as divergent.
#
# Pieces:
#   * `surface_direct_lighting_inner!` / `evaluate_material_inner!` — take the
#     BxDF carrier `get_bxdf` built for this hit and call
#     `evaluate_bsdf_spectral` / `sample_bsdf_spectral` on it directly. Sobol
#     inlined, shadow trace inlined.
#   * `vp_shade_material_kernel!` — reads `TypedHitRef{T}`, loads the hit from
#     the shared `hit_surface_queue`, looks up the
#     concrete instance via `material_of_type(materials, T, vec_idx)` (one
#     array load resolved at compile time), accumulates emission, runs DL
#     + indirect path.
#   * the "shade" pass (graph.jl) — one dispatch per queue of the
#     `MultiTypeMaterialQueue`, all in ONE pass, because they are independent:
#     each per-type kernel writes disjoint slots of the shared `next_ray_queue`
#     and atomic `pixel_L`. A pass is the unit barriers go between, so they
#     overlap on idle SMs rather than serialising.

"""Direct lighting for one surface hit: BVH light sample, BSDF evaluation on
the already-resolved `bxdf` carrier, MIS weight, shadow ray pushed inline.

Sobol samples are generated inline from `(px, py, sample_idx, base_dim+dim)`
rather than read from a pre-populated per-pixel buffer. That eliminates the
separate `vp_generate_ray_samples_kernel!` dispatch + barrier and the SOA
write/read of two `pixel_samples_direct_*` arrays per bounce — measured as
58 % of GPU time on killeroo before the change."""
@propagate_inbounds function surface_direct_lighting_inner!(
    bxdf::B,
    pixel_L,
    accel,
    media_interfaces,
    media,
    work::VPMaterialEvalWorkItem,
    materials,
    lights,
    rgb2spec_table,
    bvh_nodes,
    infinite_light_indices,
    num_infinite_lights::Int32,
    num_bvh_lights::Int32,
    num_lights::Int32,
    sobol_rng,
    sample_idx::Int32,
    camera,
    samples_per_pixel::Int32,
    do_regularize::Bool,
) where B
    num_lights < Int32(1) && return

    # Null-material boundaries never reach a per-material queue: both push
    # sites (`vp_trace_and_shade_kernel!` and delta tracking's
    # `vp_sample_medium_kernel!`) handle the medium swap inline and return, so
    # a `bxdf` here always came from a real material.

    # Inline Sobol. Dimension allocation matches the original
    # `vp_generate_ray_samples_kernel!`: 6 (camera) + 7 * depth + dim-offset;
    # direct lighting uses dim+1 (1D light select) and dim+3 (2D light pos).
    pixel_idx = work.pixel_index
    pixel_idx_0 = pixel_idx - Int32(1)
    px = u_mod(pixel_idx_0, sobol_rng.width) + Int32(1)
    py = u_div(pixel_idx_0, sobol_rng.width) + Int32(1)
    base_dim = Int32(6) + Int32(7) * work.depth
    light_select = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(1))
    u_light_x, u_light_y = sample_2d(sobol_rng, px, py, sample_idx, base_dim + Int32(3))
    u_light = Point2f(u_light_x, u_light_y)

    light_idx, light_pmf = bvh_sample_light(
        bvh_nodes, infinite_light_indices,
        num_infinite_lights, num_bvh_lights,
        work.pi, work.ns, light_select,
    )
    if light_idx < Int32(1) || light_idx > num_lights || light_pmf <= 0f0
        return
    end

    light_sample = sample_light_spectral(
        rgb2spec_table, lights, light_idx, work.pi, work.lambda, u_light,
    )

    if light_sample.pdf > 0f0 && !is_black(light_sample.Li)
        tfc = compute_texture_filter_context(work, camera, samples_per_pixel)
        regularize = do_regularize && work.any_non_specular_bounces

        # Direct call into the concrete material — Julia inlines.
        bsdf_f, bsdf_pdf = evaluate_bsdf_spectral(
            bxdf, rgb2spec_table, materials,
            work.wo, light_sample.wi, work.ns, work.dpdus, tfc, work.lambda, regularize,
        )

        if !is_black(bsdf_f)
            result = compute_direct_lighting_spectral(
                # Shadow rays leave from the terminator-safe point, not the hit.
                work.ps, work.n, work.ns, work.wo, work.beta, work.r_u, work.lambda,
                light_sample, bsdf_f, bsdf_pdf,
            )
            if result.valid
                scaled_r_l = result.r_l * light_pmf

                shadow_medium = if is_medium_transition(work.interface)
                    get_medium_index(work.interface, result.ray_direction, work.n)
                else
                    work.current_medium
                end

                # Inline shadow trace + accumulate (opt4 pattern).
                T_ray, tr_r_u, tr_r_l, visible = trace_shadow_transmittance(
                    accel, media_interfaces, media, materials, rgb2spec_table,
                    result.ray_origin, result.ray_direction, result.t_max,
                    work.lambda, shadow_medium,
                )

                if visible && !is_black(T_ray)
                    mis_weight = result.r_u * tr_r_u + scaled_r_l * tr_r_l
                    mis_denom = average(mis_weight)
                    if mis_denom > 1f-10
                        final_L = result.Ld * T_ray / mis_denom
                        if !is_black(final_L)
                            base_idx = (work.pixel_index - Int32(1)) * Int32(4)
                            accumulate_spectrum!(pixel_L, base_idx, final_L)
                        end
                    end
                end
            end
        end
    end
    return
end

"""BSDF sample for one surface hit: samples the already-resolved `bxdf`
carrier, applies Russian roulette, pushes the continuation ray."""
@propagate_inbounds function evaluate_material_inner!(
    bxdf::B,
    next_ray_queue,
    work::VPMaterialEvalWorkItem,
    materials,
    rgb2spec_table,
    max_depth::Int32,
    do_regularize::Bool,
    sobol_rng,
    sample_idx::Int32,
    camera,
    samples_per_pixel::Int32,
    rr_depth::Int32,
) where B
    # Null material → never sees a typed kernel (routed at push site).
    new_depth = work.depth + Int32(1)
    if new_depth >= max_depth
        return
    end

    pixel_idx = work.pixel_index
    pixel_idx_0 = pixel_idx - Int32(1)
    px = u_mod(pixel_idx_0, sobol_rng.width) + Int32(1)
    py = u_div(pixel_idx_0, sobol_rng.width) + Int32(1)
    base_dim = Int32(6) + Int32(7) * work.depth
    rng = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(4))
    u_x, u_y = sample_2d(sobol_rng, px, py, sample_idx, base_dim + Int32(6))
    u = Point2f(u_x, u_y)
    rr_sample = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(7))

    regularize = do_regularize && work.any_non_specular_bounces
    tfc = compute_texture_filter_context(work, camera, samples_per_pixel)

    sample = sample_bsdf_spectral(
        bxdf, rgb2spec_table, materials,
        work.wo, work.ns, work.dpdus, tfc, work.lambda, u, rng, regularize,
    )

    if sample.pdf > 0f0 && !is_black(sample.f)
        cos_theta = abs(dot(sample.wi, work.ns))
        new_beta = work.beta * sample.f * cos_theta / sample.pdf

        new_eta_scale = if is_transmissive(sample.flags)
            work.eta_scale * sample.eta * sample.eta
        else
            work.eta_scale
        end

        r_l_pdf = if sample.pdf_is_proportional
            _, p = evaluate_bsdf_spectral(
                bxdf, rgb2spec_table, materials,
                work.wo, sample.wi, work.ns, work.dpdus, tfc, work.lambda, regularize,
            )
            max(p, 1f-10)
        else
            sample.pdf
        end
        new_r_l = work.r_u / r_l_pdf

        should_continue, final_beta = russian_roulette_spectral(
            new_beta, work.r_u, new_eta_scale, new_depth, rr_sample, rr_depth,
        )

        if should_continue
            new_medium = if is_medium_transition(work.interface)
                get_medium_index(work.interface, sample.wi, work.n)
            else
                work.current_medium
            end
            offset_dir = if dot(sample.wi, work.n) > 0f0; work.n; else; -work.n; end
            ray_origin = Point3f(work.pi + offset_dir * 1f-4)
            new_ray = Raycore.Ray(o=ray_origin, d=sample.wi, t_max=Inf32, time=0f0)
            new_lambda = if sample.secondary_terminated
                terminate_secondary_wavelengths(work.lambda)
            else
                work.lambda
            end
            ray_item = VPRayWorkItem(
                new_ray, new_depth, new_lambda, work.pixel_index,
                final_beta, work.r_u, new_r_l,
                work.pi, work.ns,
                new_eta_scale,
                is_specular(sample.flags),
                work.any_non_specular_bounces || !is_specular(sample.flags),
                new_medium,
            )
            push!(next_ray_queue, ray_item)
        end
    end
    return
end

"""Per-material-type shading kernel — drains one slot of the
`MultiTypeMaterialQueue`.  Julia specialises this function separately for
each concrete `T`, so each per-type dispatch's SPIR-V contains exactly one
material's BSDF code — no `with_index` switching on the material axis.

Emission MIS still uses `with_index` on `lights` (light-type axis, separate
concern from materials)."""
@propagate_inbounds function vp_shade_material_kernel!(
    typed::TypedHitRef{T},
    hit_surface_queue,
    next_ray_queue,
    pixel_L,
    accel,
    media_interfaces,
    media,
    materials,
    lights,
    rgb2spec_table,
    bvh_nodes,
    infinite_light_indices,
    light_to_bit_trail,
    num_infinite_lights::Int32,
    num_bvh_lights::Int32,
    num_lights::Int32,
    max_depth::Int32,
    do_regularize::Bool,
    sobol_rng,
    sample_idx_ref,
    camera_ref,
    samples_per_pixel::Int32,
    rr_depth::Int32,
) where T
    sample_idx = @inbounds sample_idx_ref[Int32(1)]
    # Behind a `GPURef`, like the sample index: a moved camera is an update in
    # the run's own submission, not a new plan.
    camera = @inbounds camera_ref[Int32(1)]
    # The typed queue carries 4-byte indices; the hit payload lives once in
    # the shared hit_surface_queue.
    work = hit_surface_queue.items[typed.idx]
    wo = -work.ray.d

    # Concrete material instance — compile-time slot lookup + indexed array
    # load.  No `with_index` switch in this kernel's SPIR-V.
    mat = material_of_type(materials, T, work.material_idx.vec_idx)

    # Emission MIS for indirect rays hitting area lights now runs in its own
    # kernel (`vp_handle_emitters_kernel!` draining `hit_area_light_queue`),
    # matching pbrt-v4 wavefront. The per-material kernels stay focused on
    # BSDF sampling + direct lighting.

    # Stack-local material-eval work item.
    mat_work = VPMaterialEvalWorkItem(work, wo, work.material_idx)

    # pbrt-v4 wavefront `EvaluateMaterialAndBSDF<ConcreteMaterial>` builds the
    # BxDF once per surface hit via `Material::GetBxDF` so the subsequent
    # `Sample_f` and `f` calls operate on a small struct with already-resolved
    # spectra / textures.  Default `get_bxdf` is identity, so materials whose
    # BSDF state is already small (Diffuse: 16 B) pay nothing; Conductor's
    # 900-byte `PiecewiseLinearSpectrum{56}`-bearing struct collapses to a
    # 40-byte `ConductorEvaluated`, eliminating 8-16 redundant
    # `eval_ior_spectral` binary searches per surface hit.
    tfc_for_bxdf = compute_texture_filter_context(mat_work, camera, samples_per_pixel)
    regularize_for_bxdf = do_regularize && work.any_non_specular_bounces
    bxdf = get_bxdf(mat, rgb2spec_table, materials, tfc_for_bxdf, work.lambda, regularize_for_bxdf)

    # Direct lighting + inline shadow + accumulate (typed BSDF eval).
    surface_direct_lighting_inner!(
        bxdf,
        pixel_L, accel, media_interfaces, media,
        mat_work, materials, lights, rgb2spec_table,
        bvh_nodes, infinite_light_indices,
        num_infinite_lights, num_bvh_lights, num_lights,
        sobol_rng, sample_idx,
        camera, samples_per_pixel,
        do_regularize,
    )

    # BSDF sample + RR + push continuation (typed BSDF sample).
    evaluate_material_inner!(
        bxdf,
        next_ray_queue,
        mat_work, materials, rgb2spec_table, max_depth,
        do_regularize,
        sobol_rng, sample_idx,
        camera, samples_per_pixel,
        rr_depth,
    )
    return
end

"""Drain `hit_area_light_queue`: applies emission-MIS to each surface hit
that landed on an area-light triangle. Direct port of pbrt-v4's
"Handle emitters hit by indirect rays" kernel
(wavefront/integrator.cpp:540-572). Runs in parallel with `vp_shade_surfaces!`
because both write only to per-pixel atomic `pixel_L` accumulators (the
emitter kernel writes only `pixel_L`; the material kernels also write
`next_ray_queue` but that's disjoint from `pixel_L`)."""
@propagate_inbounds function vp_handle_emitters_kernel!(
    work,                            # VPHitAreaLightWorkItem (per-thread)
    pixel_L,
    lights, rgb2spec_table,
    bvh_nodes, light_to_bit_trail,
    num_infinite_lights::Int32, num_bvh_lights::Int32, num_lights::Int32,
)
    # Le(p, n, uv, wo)
    light_key = arealight_key(work.arealight_id)
    Le = with_index(arealight_Le, lights, light_key,
        lights, rgb2spec_table, work.wo, Vec3f(work.n), work.uv, work.lambda,
    )
    is_black(Le) && return

    contribution = work.beta * Le
    final_contrib = if work.depth == Int32(0) || work.specular_bounce
        contribution / average(work.r_u)
    else
        lightChoicePDF = bvh_pmf(
            bvh_nodes, light_to_bit_trail,
            num_infinite_lights, num_bvh_lights,
            work.prev_intr_p, work.prev_intr_n, light_flat_index(lights, light_key),
        )
        cos_theta = abs(dot(work.n, work.wo))   # wo = -ray.d → abs(dot(n, -ray.d)) = abs(dot(n, ray.d))
        lightPDF = if cos_theta > 0f0 && work.triangle_area > 0f0
            pdf_li = (work.t_hit * work.t_hit) / (cos_theta * work.triangle_area)
            lightChoicePDF * pdf_li
        else
            0f0
        end
        r_l = work.r_l * lightPDF
        mis_denom = average(work.r_u + r_l)
        if mis_denom > 1f-10
            contribution / mis_denom
        else
            contribution / average(work.r_u)
        end
    end

    base_idx = (work.pixel_index - Int32(1)) * Int32(4)
    accumulate_spectrum!(pixel_L, base_idx, final_contrib)
    return
end

