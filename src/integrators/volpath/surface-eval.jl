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

The differentials, uv derivatives and texture filter context a bump reads at a
hit, for [`bumped_shading_frame`](@ref). `face_idx` and `bary` are the hit's: a
vertex-colour bump height indexes with them.
"""
@propagate_inbounds function bump_filter_context(camera, work, geom, samples_per_pixel::Int32,
                                                 face_idx::UInt32, bary::SVector{3, Float32})
    dpdx, dpdy = surface_dp_dxy(camera, work.ray.o, work.ray.d, geom.pi, geom.n,
                                samples_per_pixel, work.depth)
    dudx, dudy, dvdx, dvdy = compute_uv_derivatives(geom.dpdu, geom.dpdv, dpdx, dpdy)
    return TextureFilterContext(geom.uv, dudx, dudy, dvdx, dvdy, face_idx, bary, geom.pi, geom.ns)
end

"""The shading frame at the hit `geom` of `primitive`, perturbed by the material's bump, if it has one."""
@propagate_inbounds function bumped_shading_frame(::Type{T}, materials, mat_idx, camera, work, geom, primitive,
                                                  samples_per_pixel::Int32, bary) where {T}
    face_idx = primitive.metadata.primitive_index
    bump_inputs() = (vp_compute_normal_derivatives(primitive)...,
                     bump_filter_context(camera, work, geom, samples_per_pixel, face_idx, SVector{3, Float32}(bary)))
    return perturbed_frame(T, materials, mat_idx, geom.ns, geom.dpdus, geom.dpdu, geom.dpdv, geom.n, bump_inputs)
end

# The kernel of a material type knows the material: only its own bump is compiled. A mix's
# component is known per hit, so its kernel asks every type in the set.
@propagate_inbounds perturbed_frame(::Type{T}, materials, idx::SetKey, ns, dpdus, dpdu, dpdv, ng, bump_inputs) where {T} =
    perturb_shading_frame_impl(material_of_type(materials, T, idx.vec_idx), materials, ns, dpdus, dpdu, dpdv, ng,
                               bump_inputs)
@propagate_inbounds perturbed_frame(::Type{<:MixMaterial}, materials, idx::SetKey, ns, dpdus, dpdu, dpdv, ng, bump_inputs) =
    get_perturbed_shading_frame(materials, idx, ns, dpdus, dpdu, dpdv, ng, bump_inputs)

"""
    shaded_material(T, materials, key, p, wo, tfc) -> SetKey

The material a hit in the queue of type `T` is shaded with: the mesh's (`key`), and for a mix the
component it picks at this hit (`resolve_mix_material`).
"""
@propagate_inbounds shaded_material(::Type, materials, key::SetKey, p::Point3f, wo::Vec3f, tfc::TextureFilterContext) = key
@propagate_inbounds shaded_material(::Type{<:MixMaterial}, materials, key::SetKey, p::Point3f, wo::Vec3f,
                                    tfc::TextureFilterContext) = resolve_mix_material(materials, key, p, wo, tfc)

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

    # Clamp to reasonable values, and zero a non-finite one, as pbrt-v4 does
    # (interaction.cpp). `clamp` passes NaN through, and a NaN derivative becomes
    # the bump map's finite-difference step, so the shifted height samples land
    # at a NaN uv, whose texel index on the GPU is garbage: the read goes past
    # the texture into whatever memory follows it.
    return map(d -> isfinite(d) ? clamp(d, -1f8, 1f8) : 0f0, (dudx, dudy, dvdx, dvdy))
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

    return TextureFilterContext(work.uv, dudx, dudy, dvdx, dvdy, work.face_idx, work.bary, work.pi, work.ns)
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
# This is the ONLY surface-shading implementation, for every traversal and for
# hits delivered by the trace and by delta tracking alike. A second copy is how
# shading drifts: a `with_index`-based one, dispatched for exactly one
# configuration (hardware RT on a scene with media), kept calling the raw
# material after every BSDF moved onto a `get_bxdf` carrier, silently matched a
# gray-Lambertian `::Material` catch-all, and rendered `medium_smoke_point` at
# 0.037 of reference on hardware while software was correct.
#
# Pieces:
#   * `surface_direct_lighting_inner!` / `evaluate_material_inner!` — take the
#     BxDF carrier `get_bxdf` built for this hit and call
#     `evaluate_bsdf_spectral` / `sample_bsdf_spectral` on it directly. Sobol
#     inlined, shadow trace inlined.
#   * `vp_shade_kernel!` — reads a `TypedHitRef{T}` from its `SlotQueue{T}`, loads the `VPSurfaceHit`
#     from the shared `surface_hits` store and its path, expands the surface,
#     looks up the concrete instance via `material_of_type(materials, T,
#     vec_idx)` (one array load resolved at compile time), accumulates emission,
#     runs DL + indirect path.
#   * the "shade" pass (graph.jl) — one dispatch per slot of the
#     `MaterialQueues`, all in ONE pass, because they are independent:
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
    # sites (`vp_trace!` and delta tracking's `sample_medium_interaction!`)
    # handle the medium swap inline and return, and a mix that resolves to no
    # material is handled likewise in `vp_shade_surface_hit!`, so a `bxdf` here
    # always came from a real material.

    # Inline Sobol. Dimension allocation matches the original
    # `vp_generate_ray_samples_kernel!`: 6 (camera) + 7 * depth + dim-offset;
    # direct lighting uses dim+1 (1D light select) and dim+3 (2D light pos).
    px, py, sample_idx = path_pixel_sample(sobol_rng, work.pixel_index, sample_idx)
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

    # The connection to the light, then one shadow ray along it. Under a refracting
    # surface (a floor under water) the sky is sampled through Snell's window, as at
    # a scatter event in the water (see `snells_window`); otherwise straight. One
    # shadow trace for both: with a call in each branch, LLVM filled the trace's
    # array arguments in 32-bit halves, and on RADV the kernel then read through a
    # garbage pointer in scenes without metaballs (a GPU page fault, device lost).
    tfc = compute_texture_filter_context(work, camera, samples_per_pixel)
    regularize = regularizes(work, media, do_regularize)
    window_medium = if is_medium_transition(work.interface)
        get_medium_index(work.interface, light_sample.wi, work.n)
    else
        work.current_medium
    end
    through_window = light_sample.pdf > 0f0 && !light_sample.is_delta && has_medium(window_medium) &&
                     Raycore.with_index(is_infinite_light, lights, flat_to_light_index(lights, light_idx))
    η = through_window ? Raycore.with_index(refracting, media, window_medium) : 0f0
    if η > 0f0
        ω_in, jacobian = snells_window(light_sample.wi, η)
        jacobian > 0f0 || return
        bsdf_f, bsdf_pdf = evaluate_bsdf_spectral(
            bxdf, rgb2spec_table, materials,
            work.wo, ω_in, work.ns, work.dpdus, tfc, work.lambda, regularize,
        )
        is_black(bsdf_f) && return
        Ld = work.beta * bsdf_f * abs(dot(ω_in, work.ns))      # times the sky, found at the end
        r_u = work.r_u * bsdf_pdf
        # The sky all the infinite lights make, over the density of their mixture.
        r_l = work.r_u * (compute_env_light_pdf(lights, light_sample.wi) * light_pmf * jacobian)
        origin, dir, t_max, medium, refract = work.ps, ω_in, 1f6, window_medium, true
    else
        (light_sample.pdf > 0f0 && !is_black(light_sample.Li)) || return
        # Direct call into the concrete material — Julia inlines.
        bsdf_f, bsdf_pdf = evaluate_bsdf_spectral(
            bxdf, rgb2spec_table, materials,
            work.wo, light_sample.wi, work.ns, work.dpdus, tfc, work.lambda, regularize,
        )
        is_black(bsdf_f) && return
        result = compute_direct_lighting_spectral(
            # Shadow rays leave from the terminator-safe point, not the hit:
            # that point lies outside the smooth surface, so only for light
            # on its outer side. Light through the surface would otherwise
            # cross the surface once more before it leaves.
            dot(light_sample.wi, work.n) > 0f0 ? work.ps : Point3f(work.pi), work.n, work.ns, work.wo,
            work.beta, work.r_u, work.lambda,
            light_sample, bsdf_f, bsdf_pdf,
        )
        result.valid || return
        Ld, r_u, r_l = result.Ld, result.r_u, result.r_l * light_pmf
        medium = if is_medium_transition(work.interface)
            get_medium_index(work.interface, result.ray_direction, work.n)
        else
            work.current_medium
        end
        origin, dir, t_max, refract = result.ray_origin, result.ray_direction, result.t_max, false
    end

    # Inline shadow trace + accumulate (opt4 pattern).
    T_ray, tr_r_u, tr_r_l, visible, out = trace_shadow_transmittance(
        accel, media_interfaces, media, materials, rgb2spec_table,
        origin, dir, t_max, work.lambda, medium, refract,
    )
    (visible && !is_black(T_ray)) || return
    mis_denom = average(r_u * tr_r_u + r_l * tr_r_l)
    mis_denom > 1f-10 || return
    # Through Snell's window the sky is where the shadow ray left (see `snells_window`).
    sky = refract ? evaluate_escaped_ray_spectral(rgb2spec_table, lights, out, work.lambda) : SpectralRadiance(1f0)
    final_L = Ld * sky * T_ray / mis_denom
    is_black(final_L) || accumulate_spectrum!(pixel_L, (work.pixel_index - Int32(1)) * Int32(4), final_L)
    return
end

"""BSDF sample for one surface hit: samples the already-resolved `bxdf`
carrier, applies Russian roulette, pushes the continuation ray."""
@propagate_inbounds function evaluate_material_inner!(
    bxdf::B,
    next_ray_queue,
    film_pdf,
    work::VPMaterialEvalWorkItem,
    materials,
    media,
    lights,
    num_infinite_lights::Int32,
    num_bvh_lights::Int32,
    rgb2spec_table,
    max_depth::Int32,
    do_regularize::Bool,
    emitters_reachable::Bool,
    sobol_rng,
    sample_idx::Int32,
    camera,
    samples_per_pixel::Int32,
    rr_depth::Int32,
) where B
    # Null material → never sees a typed kernel (routed at push site).
    new_depth = work.depth + Int32(1)
    # The deepest vertex samples the BSDF for the emission its ray reaches, if any can be
    # (`emission_reachable`, decided for the scene when the plans are built).
    (new_depth < max_depth || emitters_reachable) || return

    px, py, sample_idx = path_pixel_sample(sobol_rng, work.pixel_index, sample_idx)
    base_dim = Int32(6) + Int32(7) * work.depth
    rng = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(4))
    u_x, u_y = sample_2d(sobol_rng, px, py, sample_idx, base_dim + Int32(6))
    u = Point2f(u_x, u_y)
    rr_sample = sample_1d(sobol_rng, px, py, sample_idx, base_dim + Int32(7))

    regularize = regularizes(work, media, do_regularize)
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
            ray_origin = offset_surface_origin(work.pi,work.n,sample.wi)
            new_ray = Raycore.Ray(o=ray_origin, d=sample.wi, t_max=Inf32, time=0f0)
            new_lambda = if sample.secondary_terminated && !secondary_terminated(work.lambda)
                record_wavelengths!(film_pdf, work.pixel_index, terminate_secondary_wavelengths(work.lambda))
            else
                work.lambda
            end
            # Out of refracting water through Snell's window, straight from a non-specular
            # vertex: light sampling there reaches the sky this does, so the path keeps
            # that vertex's `r_l`, times that sampling's density for its direction
            # there (see `WINDOWED`).
            η = has_medium(work.current_medium) ? Raycore.with_index(refracting, media, work.current_medium) : 0f0
            sky, jacobian = window_sky(-work.wo, η)
            windowed_out = η > 0f0 && jacobian > 0f0 && is_transmissive(sample.flags) && is_specular(sample.flags) &&
                           work.interface.shadows > 0f0 && !work.specular_bounce && work.any_non_specular_bounces
            out_r_l = if windowed_out
                work.r_l * (infinite_light_pmf(num_infinite_lights, num_bvh_lights) *
                            compute_env_light_pdf(lights, sky) * jacobian)
            else
                new_r_l
            end
            ray_item = VPRayWorkItem(
                new_ray, new_depth, new_lambda, work.pixel_index,
                final_beta, work.r_u, out_r_l,
                work.pi, windowed_out ? WINDOWED : work.ns,
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

"""
    vp_shade_kernel!(q, surface_hits, cur, arrivals, …)

Shade the hits of one material type: drains the `SlotQueue{T}` of its slot of the
`MaterialQueues`, so each specialisation compiles exactly one material's code (and
a `MixMaterial`'s, its components').
"""
@kernel inbounds = true function vp_shade_kernel!(
    q, surface_hits, cur, arrivals,
    next_ray_queue, pixel_L, film_pdf, accel, media_interfaces, media, materials, lights,
    rgb2spec_table, bvh_nodes, infinite_light_indices, light_to_bit_trail,
    num_infinite_lights::Int32, num_bvh_lights::Int32, num_lights::Int32,
    max_depth::Int32, do_regularize::Bool, emitters_reachable::Bool, sobol_rng, sample_idx_ref, camera_ref,
    samples_per_pixel::Int32, rr_depth::Int32,
)
    j = @index(Global)
    if j <= queuelength(q)
        vp_shade_surface_hit!(q[j], surface_hits, cur, arrivals,
            next_ray_queue, pixel_L, film_pdf, accel, media_interfaces, media, materials, lights,
            rgb2spec_table, bvh_nodes, infinite_light_indices, light_to_bit_trail,
            num_infinite_lights, num_bvh_lights, num_lights,
            max_depth, do_regularize, emitters_reachable, sobol_rng, sample_idx_ref, camera_ref,
            samples_per_pixel, rr_depth)
    end
end

"""
    expanded_hit(idx, surface_hits, cur, arrivals, accel, media_interfaces)

The surface hit `idx` of `surface_hits` expanded: its path state (the round's ray, or the arrival a
medium delivered), record, primitive, barycentrics, medium interface, surface and `wo`. A shader's
values kernel and its closure kernel expand a hit by this one function, so both see one surface.
"""
@propagate_inbounds function expanded_hit(idx::Int32, surface_hits, cur, arrivals, accel, media_interfaces)
    sh = surface_hits.items[idx]
    work = sh.path > Int32(0) ? cur.items[sh.path] : arrivals.items[-sh.path]
    rec = sh.hit
    primitive = Raycore.hit_primitive(accel, rec)
    bary = Raycore.hit_barycentric(accel, rec)
    mi = media_interfaces[resolve_mi_idx(rec.custom, primitive)]
    geom = vp_compute_surface_geometry(primitive, bary, work.ray.o, work.ray.d, rec.t)
    return (; work, rec, primitive, bary, mi, geom, wo = -work.ray.d)
end

"""
    vp_shade_surface_hit!(ref, surface_hits, cur, arrivals, …)

Expand one surface hit and shade it with its material (`shaded_type(ref)`): the surface from the
hit record, the bump-perturbed shading frame, the emission of an area light it landed on
(MIS-weighted against light sampling), then the material's BSDF sampling and direct lighting
(`vp_shade_resolved_hit!`).
"""
@propagate_inbounds function vp_shade_surface_hit!(
    ref, surface_hits, cur, arrivals,
    next_ray_queue, pixel_L, film_pdf, accel, media_interfaces, media, materials, lights,
    rgb2spec_table, bvh_nodes, infinite_light_indices, light_to_bit_trail,
    num_infinite_lights::Int32, num_bvh_lights::Int32, num_lights::Int32,
    max_depth::Int32, do_regularize::Bool, emitters_reachable::Bool, sobol_rng, sample_idx_ref, camera_ref,
    samples_per_pixel::Int32, rr_depth::Int32,
)
    T = shaded_type(ref)
    # Behind `GPURef`s: a new sample index or a moved camera is an update in the
    # run's own submission, not a new plan.
    sample_idx = sample_idx_ref[Int32(1)]
    camera = camera_ref[Int32(1)]
    (; work, rec, primitive, bary, mi, geom, wo) = expanded_hit(ref.idx, surface_hits, cur, arrivals, accel,
                                                               media_interfaces)
    t_hit = rec.t
    # The mesh's material; a `MixMaterial` picks its component here, per hit.
    resolved = shaded_material(T, materials, mi.material, geom.pi, wo,
        TextureFilterContext(geom.uv, primitive.metadata.primitive_index, bary, geom.pi))

    # A mix that resolved to no material: emission, then the ray goes on, as at a
    # surface without a material in the trace.
    if !Raycore.is_valid(resolved)
        accumulate_arealight_hit!(pixel_L, lights, rgb2spec_table,
            bvh_nodes, light_to_bit_trail, num_infinite_lights, num_bvh_lights,
            primitive.metadata.arealight_id, geom.n, geom.uv, wo,
            work, work.beta, work.r_u, work.r_l, Raycore.area(primitive), t_hit)
        ray_d = -wo
        new_medium = null_crossing_medium(mi, work.medium_idx, ray_d, geom.n)
        new_ray = Raycore.Ray(o=offset_surface_origin(geom.pi, geom.n, ray_d), d=ray_d,
                              t_max=Inf32, time=0f0)
        push!(next_ray_queue, VPRayWorkItem(
            new_ray, work.depth,
            work.lambda, work.pixel_index,
            work.beta, work.r_u, work.r_l,
            work.prev_intr_p, work.prev_intr_n,
            work.eta_scale,
            work.specular_bounce, work.any_non_specular_bounces,
            new_medium))
        return nothing
    end

    # True ray differentials on camera rays, camera approximation deeper, as
    # pbrt-v4's ComputeDifferentials.
    ns_b, dpdus_b = bumped_shading_frame(T, materials, resolved, camera, work, geom, primitive,
                                         samples_per_pixel, bary)
    hit_work = VPHitSurfaceWorkItem(
        work,
        geom.pi, geom.ps, geom.n, geom.dpdu, geom.dpdv,
        ns_b, dpdus_b, cross(ns_b, dpdus_b),
        geom.uv, resolved, mi,
        primitive.metadata.primitive_index, bary,
    )
    accumulate_arealight_hit!(pixel_L, lights, rgb2spec_table,
        bvh_nodes, light_to_bit_trail, num_infinite_lights, num_bvh_lights,
        primitive.metadata.arealight_id, hit_work.n, hit_work.uv, wo,
        hit_work, hit_work.beta, hit_work.r_u, hit_work.r_l, Raycore.area(primitive), t_hit)

    # A concrete material is the queue's type, read by its slot at compile time.
    # A mix's component was picked above, so it is dispatched on its key.
    if T <: MixMaterial
        Raycore.with_index(vp_shade_resolved_hit!, materials, resolved,
            hit_work, wo, resolved,
            next_ray_queue, pixel_L, film_pdf, accel, media_interfaces, media,
            materials, lights, rgb2spec_table,
            bvh_nodes, infinite_light_indices,
            num_infinite_lights, num_bvh_lights, num_lights,
            max_depth, do_regularize, emitters_reachable, sobol_rng, sample_idx,
            camera, samples_per_pixel, rr_depth)
    else
        mat = shaded_instance(ref, materials, resolved)
        vp_shade_resolved_hit!(mat,
            hit_work, wo, resolved,
            next_ray_queue, pixel_L, film_pdf, accel, media_interfaces, media,
            materials, lights, rgb2spec_table,
            bvh_nodes, infinite_light_indices,
            num_infinite_lights, num_bvh_lights, num_lights,
            max_depth, do_regularize, emitters_reachable, sobol_rng, sample_idx,
            camera, samples_per_pixel, rr_depth, shader_geometry(T, geom, primitive,
                () -> bump_filter_context(camera, work, geom, samples_per_pixel,
                                           primitive.metadata.primitive_index, SVector{3, Float32}(bary))))
    end
    return nothing
end

"""
Record in the film that the path in slot `pixel_index` ended its secondary wavelengths: the film
divides all of the slot's radiance by these pdfs, as pbrt-v4's divides a sample's by the path's
last, so what the path gathered before counts as its first wavelength's alone too. Returns `lambda`.
"""
@propagate_inbounds function record_wavelengths!(film_pdf, pixel_index::Int32, lambda::Wavelengths)
    base = (pixel_index - Int32(1)) * Int32(4)
    film_pdf[base + Int32(1)] = lambda.pdf[1]
    film_pdf[base + Int32(2)] = lambda.pdf[2]
    film_pdf[base + Int32(3)] = lambda.pdf[3]
    film_pdf[base + Int32(4)] = lambda.pdf[4]
    return lambda
end

"""
The hit with its path's secondary wavelengths ended where `bxdf` ends them on meeting it, before
the lights are sampled there (`terminates_at_vertex`, pbrt-v4's `GetBxDF`).
"""
@propagate_inbounds function end_dispersion(bxdf, work::VPMaterialEvalWorkItem, film_pdf)
    (terminates_at_vertex(bxdf) && !secondary_terminated(work.lambda)) || return work
    return with_lambda(work, record_wavelengths!(film_pdf, work.pixel_index, terminate_secondary_wavelengths(work.lambda)))
end

@inline with_lambda(w::VPMaterialEvalWorkItem, lambda::Wavelengths) =
    VPMaterialEvalWorkItem(w.pi, w.ps, w.n, w.dpdu, w.dpdv, w.time, w.ns, w.dpdus, w.dpdvs, w.uv, w.face_idx, w.bary,
                           w.wo, w.material_idx, w.interface, lambda, w.pixel_index, w.beta, w.r_u, w.r_l, w.depth,
                           w.eta_scale, w.specular_bounce, w.any_non_specular_bounces, w.prev_intr_p, w.prev_intr_n,
                           w.current_medium)

# Shade one surface hit with the RESOLVED concrete material instance: build
# the typed BxDF, BSDF sample + RR + continuation push, then direct lighting
# (inline shadow trace). Called by the per-material shading kernel, directly or,
# for a mix's component, through `with_index`.
@propagate_inbounds function vp_shade_resolved_hit!(
    mat,
    hit_work, wo::Vec3f, resolved_mat_idx::SetKey,
    next_ray_queue, pixel_L, film_pdf, accel, media_interfaces, media,
    materials, lights, rgb2spec_table,
    bvh_nodes, infinite_light_indices,
    num_infinite_lights::Int32, num_bvh_lights::Int32, num_lights::Int32,
    max_depth::Int32, do_regularize::Bool, emitters_reachable::Bool, sobol_rng, sample_idx::Int32,
    camera, samples_per_pixel::Int32, rr_depth::Int32, bump_geometry = nothing,
)
    # The depth limit ends a path after the emission at its vertex, as pbrt-v4's VolPath:
    # the vertex before it sampled the lights AND the BSDF, and the BSDF ray's share of
    # the MIS is the light it reaches here, an area light's (`accumulate_arealight_hit!`) or
    # the material's own.
    at_limit = hit_work.depth >= max_depth
    at_limit && !emits(mat) && return
    mat_work = VPMaterialEvalWorkItem(hit_work, wo, resolved_mat_idx)

    regularize_for_bxdf = regularizes(hit_work, media, do_regularize)
    bxdf = material_bxdf(mat, rgb2spec_table, materials, mat_work, camera, samples_per_pixel,
                         hit_work.lambda, regularize_for_bxdf, bump_geometry)
    bxdf, mat_work = material_vertex(bxdf, mat_work)
    add_emission!(pixel_L, bxdf, mat_work)
    at_limit && return
    mat_work = end_dispersion(bxdf, mat_work, film_pdf)
    bxdf, mat_work = subsurface_scatter(bxdf, mat_work, accel, sample_idx)

    # BSDF sample + RR + push continuation, before the direct lighting: the
    # shadow ray is traced inside it, and anything live across that trace is
    # held through the whole traversal. Sampled after it, the BSDF was: its
    # lobes spilled around the traversal loop, and on the 7900 XTX that made
    # the hardware path twice the cost of the software one.
    evaluate_material_inner!(
        bxdf,
        next_ray_queue, film_pdf,
        mat_work, materials, media, lights, num_infinite_lights, num_bvh_lights, rgb2spec_table, max_depth,
        do_regularize, emitters_reachable,
        sobol_rng, sample_idx,
        camera, samples_per_pixel,
        rr_depth,
    )

    # Direct lighting + inline shadow trace + accumulate (typed BSDF eval).
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
    return nothing
end

"""
    accumulate_arealight_hit!(pixel_L, lights, rgb2spec_table, bvh_nodes,
        light_to_bit_trail, num_infinite_lights, num_bvh_lights,
        arealight_id, n, uv, wo, path, beta, r_u, r_l, triangle_area, t_hit)

Add the emission of a hit on an area light (`arealight_id > 0`) to the pixel,
MIS-weighted against light sampling unless the path is a camera ray or came off a
specular bounce (pbrt-v4's "Handle emitters hit by indirect rays",
wavefront/integrator.cpp:540). `path` is the ray's work item (wavelengths, depth,
previous interaction, pixel).
"""
@propagate_inbounds function accumulate_arealight_hit!(
    pixel_L, lights, rgb2spec_table,
    bvh_nodes, light_to_bit_trail, num_infinite_lights::Int32, num_bvh_lights::Int32,
    arealight_id::UInt32, n, uv::Point2f, wo::Vec3f,
    path, beta::SpectralRadiance, r_u::SpectralRadiance, r_l::SpectralRadiance,
    triangle_area::Float32, t_hit::Float32,
)
    arealight_id > UInt32(0) || return nothing
    light_key = arealight_key(arealight_id)
    Le = with_index(arealight_Le, lights, light_key,
        lights, rgb2spec_table, wo, Vec3f(n), uv, path.lambda,
    )
    is_black(Le) && return nothing
    contribution = beta * Le
    final_contrib = if path.depth == Int32(0) || path.specular_bounce
        contribution / average(r_u)
    else
        lightChoicePDF = bvh_pmf(
            bvh_nodes, light_to_bit_trail,
            num_infinite_lights, num_bvh_lights,
            path.prev_intr_p, path.prev_intr_n, light_flat_index(lights, light_key),
        )
        cos_theta = abs(dot(Vec3f(n), wo))
        lightPDF = if cos_theta > 0f0 && triangle_area > 0f0
            pdf_li = (t_hit * t_hit) / (cos_theta * triangle_area)
            lightChoicePDF * pdf_li
        else
            0f0
        end
        mis_denom = average(r_u + r_l * lightPDF)
        if mis_denom > 1f-10
            contribution / mis_denom
        else
            contribution / average(r_u)
        end
    end
    base_idx = (path.pixel_index - Int32(1)) * Int32(4)
    accumulate_spectrum!(pixel_L, base_idx, final_contrib)
    return nothing
end
