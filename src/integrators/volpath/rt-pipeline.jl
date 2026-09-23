# Hikari VolPath: RT pipeline (raygen/closesthit/miss) trace+shade variant
#
# This follows the pbrt-v4 OptiX pattern (not the lazy "everything in raygen"
# variant): the raygen only calls `traceRay`; all post-hit shading work
# (geometry, bump, MixMaterial, push to `per_material_queue`) runs in the
# CLOSEST-HIT shader, and the escape case (push to `escaped_queue` or
# `medium_sample_queue`) runs in the MISS shader.
#
# This is the structure that lets NVIDIA's driver do anything useful for us
# at the RT layer — coherence reordering (SER, when we add it), per-material
# closest-hit shaders via SBT, etc. all hook in at the chit boundary.
#
# Both `closest_hit` and `miss` receive the same BDA argument signature as
# the raygen (queues, materials, etc.) — see Lava's
# `RayTracingPipeline(... chit_miss_take_args=true)` switch.  Both stages
# look up their work item via `work_queue.items[rt_launch_id_x() + 1]`.

# The DEVICE-side intrinsics are Lava's — they are what a shader body calls, and
# the compiler is what lowers them. The PIPELINE that dispatches them is Mantle's,
# because building one needs a device.
#
# That split is the 2026-08-27 move in one import block: everything a kernel says
# comes from the compiler, everything that runs one comes from the runtime.
# DELETED in phase 1.5: see Mantle/docs/mantle-owns-it.md
#
# `import Lava` and the thirteen `lava_rt_*` intrinsics. This was the ONLY
# reason a renderer that reaches the GPU through Mantle for everything else
# named a backend at load time — and once Lava depended on Vulkan again, the
# reason Hikari stopped loading on any machine without a Vulkan driver.
#
# The hardware path (`hw-rt.jl`) never needed it: it imports `Mantle: HWTLAS,
# AdaptedAccel` and nothing else, and it is what runs on Metal. Phase 2.1 gives
# these intrinsics vendor-free names in KernelInterface.
import Mantle
import Mantle: RayTracingPipeline, trace_rays_indirect!

# ─────────────────────────────────────────────────────────────────────────────
# Raygen: pull the ray work item, fire `traceRay`, let chit/miss do the work.
# ─────────────────────────────────────────────────────────────────────────────

@propagate_inbounds function vp_trace_raygen(
    work_queue,                                  # WorkQueue{VPRayWorkItem}
    next_ray_queue, escaped_queue, medium_sample_queue,
    per_material_queue,
    hit_surface_queue,
    hit_area_light_queue,
    pixel_L,
    accel,
    media_interfaces, media, materials, lights,
    rgb2spec_table,
    bvh_nodes, infinite_light_indices, light_to_bit_trail,
    num_infinite_lights::Int32, num_bvh_lights::Int32, num_lights::Int32,
    max_depth::Int32, do_regularize::Bool,
    sobol_rng, sample_idx_ref,
    camera_ref,
    samples_per_pixel::Int32,
    rr_depth::Int32,
)
    sample_idx = @inbounds sample_idx_ref[Int32(1)]
    lid = rt_launch_id_x()
    @inbounds work = work_queue.items[Int(lid) + 1]
    ray = work.ray

    # SER pattern (pbrt-v4 OptiX):
    #   1. hitObjectTraceRayNV fills the implicit HitObject WITHOUT firing chit.
    #   2. reorderThreadWithHitObjectNV reshuffles the warp so threads with
    #      similar work (same hit primitive / instance / miss) end up adjacent.
    #   3. hitObjectExecuteShaderNV finally invokes the chit (or miss) — but
    #      now with the reordered warp, so vp_closesthit_shade runs coherently.
    # On hardware/drivers that do not support SER, the SPIR-V emitter falls
    # back to the implicit-trace path automatically (no SER opcodes emitted).
    rt_hit_object_trace_ray(
        UInt32(0),                    # ray flags
        UInt32(0xFF),                 # cull mask
        UInt32(0),                    # SBT record offset
        UInt32(0),                    # SBT record stride
        UInt32(0),                    # miss index
        Float32(ray.o[1]), Float32(ray.o[2]), Float32(ray.o[3]), Float32(ray.t_min),
        Float32(ray.d[1]), Float32(ray.d[2]), Float32(ray.d[3]), Float32(ray.t_max),
    )
    rt_reorder_thread()
    rt_hit_object_execute_shader()
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Closesthit: read the work item back, do the surface-shade body of
# `vp_trace_and_shade_kernel!`'s non-medium branch (or the medium-survives-
# to-surface branch of the medium case), push to the appropriate queue.
# ─────────────────────────────────────────────────────────────────────────────

@propagate_inbounds function vp_closesthit_shade(
    work_queue,
    next_ray_queue, escaped_queue, medium_sample_queue,
    per_material_queue,
    hit_surface_queue,
    hit_area_light_queue,
    pixel_L,
    accel,
    media_interfaces, media, materials, lights,
    rgb2spec_table,
    bvh_nodes, infinite_light_indices, light_to_bit_trail,
    num_infinite_lights::Int32, num_bvh_lights::Int32, num_lights::Int32,
    max_depth::Int32, do_regularize::Bool,
    sobol_rng, sample_idx_ref,
    camera_ref,
    samples_per_pixel::Int32,
    rr_depth::Int32,
)
    sample_idx = @inbounds sample_idx_ref[Int32(1)]
    camera = @inbounds camera_ref[Int32(1)]
    lid = rt_launch_id_x()
    @inbounds work = work_queue.items[Int(lid) + 1]

    # Hit info from RT intrinsics.
    t_hit = rt_ray_tmax()
    prim_id = rt_primitive_id()
    inst_id = rt_instance_id()
    inst_custom_idx = rt_instance_custom_index()
    bu = rt_hit_bary_u()
    bv = rt_hit_bary_v()
    bary = SVector{3,Float32}(1f0 - bu - bv, bu, bv)

    # Look up the triangle from the per-instance flat array.
    @inbounds tri_idx = Int(accel.offsets[inst_id + UInt32(1)]) + Int(prim_id) + 1
    @inbounds primitive = accel.triangles[tri_idx]

    # ─── Medium branch: same routing as `vp_trace_and_shade_kernel!`'s top ───
    if has_medium(work.medium_idx)
        mi_idx = resolve_mi_idx(accel, inst_custom_idx, primitive)
        mi = media_interfaces[mi_idx]
        mat_idx = mi.material

        geom = vp_compute_surface_geometry(primitive, bary, work.ray.o, work.ray.d, t_hit)

        tfc_bump = bump_filter_context(camera, work, geom, samples_per_pixel)
        dndu, dndv = vp_compute_normal_derivatives(primitive)
        ns_b, dpdus_b = get_perturbed_shading_frame(materials, mat_idx,
                                                   geom.ns, geom.dpdus,
                                                   geom.dpdu, geom.dpdv,
                                                   dndu, dndv, geom.n, tfc_bump)

        dpdvs_b = cross(ns_b, dpdus_b)

        push!(medium_sample_queue, VPMediumSampleWorkItem(
            work, t_hit,
            geom.pi, geom.n, geom.dpdu, geom.dpdv,
            ns_b, dpdus_b, dpdvs_b,
            geom.uv, mat_idx, mi,
            primitive.metadata.primitive_index, SVector{3,Float32}(bary),
            primitive.metadata.arealight_flat_idx, Raycore.area(primitive)
        ))
        return nothing
    end

    # ─── Non-medium: surface shade ───
    # NOTE on alpha test: the compute+ray-query kernel does a 16-iteration
    # alpha-test loop here.  In the RT pipeline that belongs in an `anyhit`
    # shader (the GPU re-invokes anyhit per intersection candidate and we
    # call `rt_ignore_intersection()` to skip).  This first
    # implementation does NOT have an anyhit; scenes with alpha textures
    # will treat them as opaque under HW RT.  Add anyhit as a follow-up
    # when the basic chit-shading variant is shown to win.
    mi_idx = resolve_mi_idx(accel, inst_custom_idx, primitive)
    mi = media_interfaces[mi_idx]
    mat_idx = mi.material

    geom = vp_compute_surface_geometry(primitive, bary, work.ray.o, work.ray.d, t_hit)
    wo = -work.ray.d
    resolved_mat_idx = resolve_mix_material(materials, mat_idx, geom.pi, wo, geom.uv)

    # Null-material boundary.
    if !Raycore.is_valid(resolved_mat_idx) && is_medium_transition(mi)
        ray_d = -wo
        new_medium = get_medium_index(mi, ray_d, geom.n)
        offset_dir = if dot(ray_d, geom.n) > 0f0; geom.n; else; -geom.n; end
        ray_origin = Point3f(geom.pi + offset_dir * 1f-4)
        new_ray = Raycore.Ray(o=ray_origin, d=ray_d, t_max=Inf32, time=0f0)
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

    # True ray differentials on camera rays, camera approximation deeper —
    # pbrt-v4 ComputeDifferentials branches the same way (interaction.cpp:48).
    tfc_bump = bump_filter_context(camera, work, geom, samples_per_pixel)
    dndu, dndv = vp_compute_normal_derivatives(primitive)
    ns_b, dpdus_b = get_perturbed_shading_frame(materials, resolved_mat_idx,
                                               geom.ns, geom.dpdus,
                                               geom.dpdu, geom.dpdv,
                                               dndu, dndv, geom.n, tfc_bump)

    dpdvs_b = cross(ns_b, dpdus_b)

    hit_work = VPHitSurfaceWorkItem(
        work,
        geom.pi, geom.n, geom.dpdu, geom.dpdv,
        ns_b, dpdus_b, dpdvs_b,
        geom.uv, resolved_mat_idx, mi,
        primitive.metadata.primitive_index, SVector{3,Float32}(bary),
    )

    enqueue_after_intersection!(per_material_queue, hit_area_light_queue, materials,
        hit_surface_queue, hit_work,
        primitive.metadata.arealight_flat_idx, Raycore.area(primitive), t_hit)
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Miss: push to the appropriate queue based on whether the ray is in a medium.
# ─────────────────────────────────────────────────────────────────────────────

@propagate_inbounds function vp_miss_escape(
    work_queue,
    next_ray_queue, escaped_queue, medium_sample_queue,
    per_material_queue,
    hit_surface_queue,
    hit_area_light_queue,
    pixel_L,
    accel,
    media_interfaces, media, materials, lights,
    rgb2spec_table,
    bvh_nodes, infinite_light_indices, light_to_bit_trail,
    num_infinite_lights::Int32, num_bvh_lights::Int32, num_lights::Int32,
    max_depth::Int32, do_regularize::Bool,
    sobol_rng, sample_idx_ref,
    camera_ref,
    samples_per_pixel::Int32,
    rr_depth::Int32,
)
    sample_idx = @inbounds sample_idx_ref[Int32(1)]
    lid = rt_launch_id_x()
    @inbounds work = work_queue.items[Int(lid) + 1]
    if has_medium(work.medium_idx)
        push!(medium_sample_queue, VPMediumSampleWorkItem(work))
    else
        push!(escaped_queue, VPEscapedRayWorkItem(work))
    end
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Per-material chit closures (SBT slot per concrete material type)
# ─────────────────────────────────────────────────────────────────────────────
#
# Each `VPClosesthitTyped{T}` is a callable singleton that runs as a closest
# hit shader for one concrete material `T`. Lava compiles one SPIR-V chit
# per `T`; the SBT has one hit group per chit; each HWTLAS instance's
# `instanceShaderBindingTableRecordOffset` selects the slot matching its
# mesh's material type so the GPU dispatches the right chit directly — no
# with_index switch inside.
#
# The chit body merges the work that used to be split across three kernels:
#   1. `vp_closesthit_shade`         — routing + geometry + null boundary
#   2. `vp_handle_emitters_kernel!`  — area-light emission MIS
#   3. `vp_shade_material_kernel!`   — DL + BSDF sample + RR + continuation
# so the chit fully shades the hit in one pass, eliminating the per-material
# / emitter post-hoc indirect dispatches.

struct VPClosesthitTyped{T} end

@propagate_inbounds function (::VPClosesthitTyped{T})(
    work_queue,
    next_ray_queue, escaped_queue, medium_sample_queue,
    per_material_queue,                  # legacy, unused on per-mat chit path
    hit_surface_queue,                   # legacy, unused on per-mat chit path
    hit_area_light_queue,                # legacy, unused on per-mat chit path
    pixel_L,
    accel,
    media_interfaces, media, materials, lights,
    rgb2spec_table,
    bvh_nodes, infinite_light_indices, light_to_bit_trail,
    num_infinite_lights::Int32, num_bvh_lights::Int32, num_lights::Int32,
    max_depth::Int32, do_regularize::Bool,
    sobol_rng, sample_idx_ref,
    camera_ref,
    samples_per_pixel::Int32,
    rr_depth::Int32,
) where {T}
    sample_idx = @inbounds sample_idx_ref[Int32(1)]
    camera = @inbounds camera_ref[Int32(1)]
    lid = rt_launch_id_x()
    @inbounds work = work_queue.items[Int(lid) + 1]

    t_hit = rt_ray_tmax()
    prim_id = rt_primitive_id()
    inst_id = rt_instance_id()
    inst_custom_idx = rt_instance_custom_index()
    bu = rt_hit_bary_u()
    bv = rt_hit_bary_v()
    bary = SVector{3,Float32}(1f0 - bu - bv, bu, bv)

    @inbounds tri_idx = Int(accel.offsets[inst_id + UInt32(1)]) + Int(prim_id) + 1
    @inbounds primitive = accel.triangles[tri_idx]

    # Medium branch — defer to compute kernel that drains medium_sample_queue.
    if has_medium(work.medium_idx)
        mi_idx = resolve_mi_idx(accel, inst_custom_idx, primitive)
        mi = media_interfaces[mi_idx]
        mat_idx = mi.material

        geom = vp_compute_surface_geometry(primitive, bary, work.ray.o, work.ray.d, t_hit)

        tfc_bump = bump_filter_context(camera, work, geom, samples_per_pixel)
        dndu, dndv = vp_compute_normal_derivatives(primitive)
        ns_b, dpdus_b = get_perturbed_shading_frame(materials, mat_idx,
                                                   geom.ns, geom.dpdus,
                                                   geom.dpdu, geom.dpdv,
                                                   dndu, dndv, geom.n, tfc_bump)
        dpdvs_b = cross(ns_b, dpdus_b)

        push!(medium_sample_queue, VPMediumSampleWorkItem(
            work, t_hit,
            geom.pi, geom.n, geom.dpdu, geom.dpdv,
            ns_b, dpdus_b, dpdvs_b,
            geom.uv, mat_idx, mi,
            primitive.metadata.primitive_index, SVector{3,Float32}(bary),
            primitive.metadata.arealight_flat_idx, Raycore.area(primitive)
        ))
        return nothing
    end

    # Non-medium surface hit
    mi_idx = resolve_mi_idx(accel, inst_custom_idx, primitive)
    mi = media_interfaces[mi_idx]
    mat_idx = mi.material

    geom = vp_compute_surface_geometry(primitive, bary, work.ray.o, work.ray.d, t_hit)
    wo = -work.ray.d
    resolved_mat_idx = resolve_mix_material(materials, mat_idx, geom.pi, wo, geom.uv)

    # Null-material boundary (medium transition surface).
    if !Raycore.is_valid(resolved_mat_idx) && is_medium_transition(mi)
        ray_d = -wo
        new_medium = get_medium_index(mi, ray_d, geom.n)
        offset_dir = if dot(ray_d, geom.n) > 0f0; geom.n; else; -geom.n; end
        ray_origin = Point3f(geom.pi + offset_dir * 1f-4)
        new_ray = Raycore.Ray(o=ray_origin, d=ray_d, t_max=Inf32, time=0f0)
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

    # True ray differentials on camera rays, camera approximation deeper —
    # see the chit-shade comment above.
    tfc_bump = bump_filter_context(camera, work, geom, samples_per_pixel)
    dndu, dndv = vp_compute_normal_derivatives(primitive)
    ns_b, dpdus_b = get_perturbed_shading_frame(materials, resolved_mat_idx,
                                               geom.ns, geom.dpdus,
                                               geom.dpdu, geom.dpdv,
                                               dndu, dndv, geom.n, tfc_bump)
    dpdvs_b = cross(ns_b, dpdus_b)

    hit_work = VPHitSurfaceWorkItem(
        work,
        geom.pi, geom.n, geom.dpdu, geom.dpdv,
        ns_b, dpdus_b, dpdvs_b,
        geom.uv, resolved_mat_idx, mi,
        primitive.metadata.primitive_index, SVector{3,Float32}(bary),
    )

    # Area-light emission MIS (replaces vp_handle_emitters_kernel for this hit).
    arealight_flat_idx = primitive.metadata.arealight_flat_idx
    if arealight_flat_idx > UInt32(0)
        light_idx = flat_to_light_index(lights, Int32(arealight_flat_idx))
        Le = with_index(arealight_Le, lights, light_idx,
            lights, rgb2spec_table, wo, Vec3f(hit_work.n), hit_work.uv, hit_work.lambda,
        )
        if !is_black(Le)
            contribution = hit_work.beta * Le
            final_contrib = if hit_work.depth == Int32(0) || hit_work.specular_bounce
                contribution / average(hit_work.r_u)
            else
                lightChoicePDF = bvh_pmf(
                    bvh_nodes, light_to_bit_trail,
                    num_infinite_lights, num_bvh_lights,
                    hit_work.prev_intr_p, hit_work.prev_intr_n, Int32(arealight_flat_idx),
                )
                cos_theta = abs(dot(hit_work.n, wo))
                triangle_area = Raycore.area(primitive)
                lightPDF = if cos_theta > 0f0 && triangle_area > 0f0
                    pdf_li = (t_hit * t_hit) / (cos_theta * triangle_area)
                    lightChoicePDF * pdf_li
                else
                    0f0
                end
                r_l = hit_work.r_l * lightPDF
                mis_denom = average(hit_work.r_u + r_l)
                if mis_denom > 1f-10
                    contribution / mis_denom
                else
                    contribution / average(hit_work.r_u)
                end
            end
            base_idx = (hit_work.pixel_index - Int32(1)) * Int32(4)
            accumulate_spectrum!(pixel_L, base_idx, final_contrib)
        end
    end

    # Per-material shading. The SBT routes by the MESH's material type T.
    # For concrete materials, `resolved_mat_idx` still points into T's slot
    # and `material_of_type` is a compile-time slot lookup. For MixMaterial
    # meshes the SBT slot is Mix but `resolve_mix_material` above re-pointed
    # `resolved_mat_idx` at one of the two SUB-materials — reading that index
    # through T would fetch a MixMaterial at the sub-material's vec_idx
    # (wrong material entirely; mat_mix_light_point pinned tile=0.75). Those
    # hits dispatch dynamically on the resolved key via `with_index`, the
    # same routing `enqueue_after_intersection!` does for the compute path.
    if T <: MixMaterial
        Raycore.with_index(vp_shade_resolved_hit!, materials, resolved_mat_idx,
            hit_work, wo, resolved_mat_idx,
            next_ray_queue, pixel_L, accel, media_interfaces, media,
            materials, lights, rgb2spec_table,
            bvh_nodes, infinite_light_indices,
            num_infinite_lights, num_bvh_lights, num_lights,
            max_depth, do_regularize, sobol_rng, sample_idx,
            camera, samples_per_pixel, rr_depth)
    else
        mat = material_of_type(materials, T, resolved_mat_idx.vec_idx)
        vp_shade_resolved_hit!(mat,
            hit_work, wo, resolved_mat_idx,
            next_ray_queue, pixel_L, accel, media_interfaces, media,
            materials, lights, rgb2spec_table,
            bvh_nodes, infinite_light_indices,
            num_infinite_lights, num_bvh_lights, num_lights,
            max_depth, do_regularize, sobol_rng, sample_idx,
            camera, samples_per_pixel, rr_depth)
    end
    return nothing
end

# Shade one surface hit with the RESOLVED concrete material instance: build
# the typed BxDF, run direct lighting (inline shadow trace) and BSDF sample +
# RR + continuation push. Shared by the compile-time path (SBT slot == hit
# material type) and the MixMaterial chit's runtime `with_index` dispatch.
@propagate_inbounds function vp_shade_resolved_hit!(
    mat,
    hit_work, wo::Vec3f, resolved_mat_idx::SetKey,
    next_ray_queue, pixel_L, accel, media_interfaces, media,
    materials, lights, rgb2spec_table,
    bvh_nodes, infinite_light_indices,
    num_infinite_lights::Int32, num_bvh_lights::Int32, num_lights::Int32,
    max_depth::Int32, do_regularize::Bool, sobol_rng, sample_idx::Int32,
    camera, samples_per_pixel::Int32, rr_depth::Int32,
)
    mat_work = VPMaterialEvalWorkItem(hit_work, wo, resolved_mat_idx)

    tfc_for_bxdf = compute_texture_filter_context(mat_work, camera, samples_per_pixel)
    regularize_for_bxdf = do_regularize && hit_work.any_non_specular_bounces
    bxdf = get_bxdf(mat, rgb2spec_table, materials, tfc_for_bxdf, hit_work.lambda, regularize_for_bxdf)

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
    return nothing
end

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline + dispatch
# ─────────────────────────────────────────────────────────────────────────────

# Build the chit tuple matching the materials set's static type tuple. Slot
# order MATCHES `materials.data` order, which is what `material_type_slot`
# returns for each concrete material at HWTLAS push! time.
@inline @generated function _build_per_material_chits(
    ::Raycore.StaticMultiTypeSet{Data},
) where {Data}
    exprs = Expr[]
    for vec_type in Data.parameters
        T = eltype(vec_type)
        push!(exprs, :(VPClosesthitTyped{$T}()))
    end
    return Expr(:tuple, exprs...)
end

"""
    material_type_slot(materials, ::Type{M}) -> UInt32

Return the SBT hit-group slot for material type `M` in the per-material RT
pipeline. Slot is the 0-based index of `M` in the materials set's static
type tuple — the same order `_build_per_material_chits` uses.
"""
@inline @generated function material_type_slot(
    materials::Raycore.StaticMultiTypeSet{Data},
    ::Type{M},
) where {Data, M}
    for (i, vec_type) in enumerate(Data.parameters)
        if eltype(vec_type) === M
            return :(UInt32($(i - 1)))
        end
    end
    return :(error($("material_type_slot: type $M not present in materials set")))
end

# Per-scene pipeline cache, keyed by the materials set's concrete type tuple.
# Different scenes share a pipeline iff their materials lists have the same
# concrete-type signature.
const _VP_RT_PIPELINES = IdDict{Any, Any}()

function vp_rt_pipeline(materials::Raycore.StaticMultiTypeSet{Data}) where {Data}
    p = get(_VP_RT_PIPELINES, Data, nothing)
    if p === nothing
        chits = _build_per_material_chits(materials)
        p = RayTracingPipeline(
            raygen      = vp_trace_raygen,
            closest_hit = chits,
            miss        = vp_miss_escape,
            payload_type = :f32_7,
            chit_miss_take_args = true,
        )
        _VP_RT_PIPELINES[Data] = p
    end
    return p
end

"""
The trace stage on the hardware ray-tracing pipeline.

A `custom!` pass rather than a `compute!` one, and that is the whole difference:
`vkCmdTraceRaysIndirect` with a shader binding table is not a dispatch, so there
is no kernel and no ndrange for the graph to record. What it reads and writes is
the same either way, which is why it is declared with the same
[`trace_uses!`](@ref) as the compute form — the ordering the graph derives does
not depend on which pipeline does the tracing.

The body reads the scene out of the `Ref`s at record time, so a camera move or a
new sample index does not rebuild the plan.
"""
function trace_pass!(g, accel_t::Mantle.AdaptedAccel, state::VolPathState, refs,
                     cur::WorkQueue, nxt::WorkQueue)
    # A hardware accel does not imply a ray-tracing PIPELINE. Vulkan hands an
    # SBT to the driver; Metal traverses through a linked function called from
    # the shading kernel, which is the generic compute path below this method.
    # Both are hardware traversal — the question is who invokes the hit shaders,
    # and that is a backend capability rather than a property of the accel type.
    #
    # …and a ray-tracing pipeline needs at least one HIT GROUP, which a scene
    # with no materials cannot give it. That is not a degenerate case: an
    # `LScene` holding only `scatter!` and `lines!` has its geometry drawn by
    # the rasteriser and its sky by a light, and so does a traced scene the
    # moment `setrasterize!` hands its last mesh to the other renderer. Nothing
    # can be hit, so the generic compute path traces the (empty) structure and
    # takes every ray to the miss shader — which is the whole picture there.
    if !Mantle.supports_rt_pipeline(accel_t) || isempty(refs.materials)
        return invoke(trace_pass!,
                      Tuple{Any, Any, VolPathState, Any, WorkQueue, WorkQueue},
                      g, accel_t, state, refs, cur, nxt)
    end
    # `trace!`, not `custom!`. The body used to be a closure that opened the
    # queue itself, packed its own arguments into per-frame scratch and called
    # `trace_rays_indirect!`. That is what `custom!` is for — work the graph
    # declares but does not model — and tracing is not that: it is one launch
    # with arguments, which is what `Trace` says. Two things follow from the
    # change and both are the point.
    #
    #   * The arguments live in the plan's argument memory at a fixed offset, so
    #     a hardware-RT plan can be RECORDED. Under `custom!` the address baked
    #     into the captured command buffer pointed into a scratch slab the pool
    #     later rewound.
    #   * `refs.sample_idx` goes in as a `Mantle.GPURef`, so the raygen holds its
    #     ADDRESS and a new sample index reaches a plan recorded once — the old
    #     body packed the number by hand, per run.
    #
    # `vp_rt_pipeline` is resolved here rather than per record because the
    # per-material closest-hit set decides the shader binding table: a different
    # material set is a different pipeline, and that is a recompile either way.
    Mantle.trace!(g,
                  vp_rt_pipeline(refs.materials),
                  refs.accel,
                  (cur, nxt,
                   state.escaped_queue,
                   state.medium_sample_queue,
                   state.per_material_queue,
                   state.hit_surface_queue,
                   state.hit_area_light_queue,
                   state.pixel_L,
                   refs.accel, refs.media_interfaces, refs.media, refs.materials, refs.lights,
                   state.rgb2spec_table,
                   state.bvh_nodes,
                   state.infinite_light_indices,
                   state.light_to_bit_trail,
                   state.num_infinite_lights,
                   state.num_bvh_lights,
                   state.num_lights,
                   state.max_depth,
                   refs.regularize,
                   state.sobol_rng, refs.sample_idx,
                   refs.camera, refs.samples_per_pixel,
                   state.rr_depth),
                  # The CEILING, like every other queue dispatch in `graph.jl`. Without
                  # it `bakedrange` builds a `DeferredRange`, whose `ndrangeof` calls
                  # `awaitwrites` — a full device synchronise, on the host, before this
                  # dispatch every single time. On a KernelAbstractions backend that
                  # was measured at 320 syncs and 0.585 s of a 0.602 s frame; the
                  # traversal itself was never the cost.
                  #
                  # Over-dispatching is defined to be a no-op for the surplus threads:
                  # every kernel dispatched over a `DeviceRange` bounds-check itself,
                  # repo-wide, which is what makes the ceiling safe. This was the one
                  # queue dispatch in the integrator that still read its count on the
                  # host.
                  Mantle.DeviceRange(cur.size; max = Int(cur.capacity)))
end
