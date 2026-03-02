# Hikari Hardware RT Integration for Lava.jl
#
# This file provides HWTLAS — a drop-in replacement for Raycore.TLAS that uses
# Vulkan's VK_KHR_ray_tracing_pipeline for hardware BVH traversal.
#
# Usage (drop-in TLAS replacement):
#   hwtlas = HWTLAS()
#   scene = Hikari.Scene(pairs; accel=hwtlas)
#   # or for RayMakie:
#   scene = Hikari.Scene(; backend=LavaBackend(), accel=HWTLAS())
#
# The HWTLAS implements the same push!/sync! API as Raycore.TLAS.
# On push!, a Vulkan BLAS is built immediately from the mesh geometry.
# On sync!(), a Vulkan TLAS is built from all accumulated BLASes.
#
# When a scene has an HWTLAS, vp_trace_rays! and vp_trace_shadow_rays!
# automatically dispatch to hardware implementations via the adapted accel type.
#
# Current limitations:
# - Alpha transparency (cutout textures) is NOT fully supported — the any-hit
#   shader handles medium transitions (glass) but not texture-based alpha.
#   Full alpha support requires texture sampling in the any-hit shader (future work).

using Lava: RTRay, RTHitResult, LavaArray, LavaBackend, LavaBLAS, LavaTLAS,
            lava_global_invocation_id_x, trace_closest_hits!, trace_closest_hits_indirect!,
            trace_closest_hits_anyhit!, trace_closest_hits_anyhit_indirect!,
            set_anyhit_pipeline!,
            _lava_rt_ignore_intersection, _lava_rt_terminate_ray,
            lava_rt_primitive_id, lava_rt_instance_custom_index,
            lava_rt_launch_id_x, _lava_rt_payload_store_f32_at, _lava_rt_payload_load_f32_at,
            _lava_rt_trace_ray,
            vk_flush!, build_blas, build_tlas, _mat4_to_vk_transform

using GeometryBasics: decompose, decompose_normals, TriangleFace

const Mat4f = SMatrix{4, 4, Float32, 16}

# Override Scene's _default_accel for LavaBackend + hw_accel=true
_default_accel(::LavaBackend, ::Val{true}) = HWTLAS()

# ══════════════════════════════════════════════════════════════════════════════
# HWTLAS — Hardware-accelerated TLAS (no CPU BVH)
# ══════════════════════════════════════════════════════════════════════════════

"""
    HWTLAS

Hardware-accelerated TLAS using Vulkan's `VK_KHR_ray_tracing_pipeline`.
Drop-in replacement for `Raycore.TLAS` — same `push!/sync!` API.

On `push!()`, builds a Vulkan BLAS immediately from the mesh geometry.
On `sync!()`, builds the Vulkan TLAS from all accumulated BLASes.
No CPU BVH is ever constructed.

# Usage
```julia
hwtlas = HWTLAS()
scene = Hikari.Scene(pairs; accel=hwtlas)
vp = Hikari.VolPath(samples=10, max_depth=8)
vp(scene, gpu_film, camera)  # automatically uses HW RT
```
"""
mutable struct HWTLAS
    # ── Geometry storage (accumulated on push!) ──
    blas_list::Vector{LavaBLAS}             # One per unique geometry
    blas_triangles::Vector{Vector{Any}}     # Per-BLAS triangle data (for shading)
    blas_offsets::Vector{UInt32}            # Per-BLAS offset into flat triangle array

    # ── Instance management ──
    instance_blas_indices::Vector{Int}       # Which BLAS each instance references (1-based)
    instance_transforms::Vector{NTuple{12,Float32}}  # VkTransformMatrixKHR per instance
    instance_custom_indices::Vector{UInt32}  # BLAS index (0-based) per instance

    handle_to_range::Dict{TLASHandle, UnitRange{Int}}
    deleted_handles::Set{TLASHandle}
    next_handle_id::UInt32

    # ── Bounding box ──
    root_aabb::Bounds3

    # ── Built on sync! ──
    hw_tlas::Union{Nothing, LavaTLAS}
    hw_accel::Union{Nothing, Lava.HardwareAccel}  # Wraps hw_tlas + RT pipeline
    tri_gpu::Union{Nothing, LavaArray}             # All triangles (flat) on GPU
    off_gpu::Union{Nothing, LavaArray{UInt32, 1}}  # BLAS offsets on GPU

    dirty::Bool

    # ── Pre-allocated ray trace buffers (reused across frames) ──
    primary_ray_buf::Union{Nothing, LavaArray{RTRay, 1}}
    primary_result_buf::Union{Nothing, LavaArray{RTHitResult, 1}}
    shadow_states::Union{Nothing, LavaArray}
    shadow_ray_buf::Union{Nothing, LavaArray{RTRay, 1}}
    shadow_result_buf::Union{Nothing, LavaArray{RTHitResult, 1}}
    shadow_active_counter::Union{Nothing, LavaArray{Int32, 1}}
end

"""
    HWTLAS()

Create an empty HWTLAS. Use `push!` to add geometry (builds Vulkan BLAS
immediately), then `sync!` to build the Vulkan TLAS.
"""
HWTLAS() = HWTLAS(
    LavaBLAS[], Vector{Any}[], UInt32[],
    Int[], NTuple{12,Float32}[], UInt32[],
    Dict{TLASHandle, UnitRange{Int}}(), Set{TLASHandle}(), UInt32(1),
    Bounds3(),
    nothing, nothing, nothing, nothing,
    true,
    nothing, nothing, nothing, nothing, nothing, nothing,
)

# ── push! — build Vulkan BLAS immediately ──

function Base.push!(hwtlas::HWTLAS, geometry, material_idx; arealight_indices=nothing)
    return push!(hwtlas, Raycore.Instance(geometry; metadata=material_idx); arealight_indices)
end

function Base.push!(hwtlas::HWTLAS, geometry, material_idx, transform::Mat4f; arealight_indices=nothing)
    return push!(hwtlas, Raycore.Instance(geometry, transform, material_idx); arealight_indices)
end

function Base.push!(hwtlas::HWTLAS, geometry, material_idx, transforms::AbstractVector{Mat4f}; arealight_indices=nothing)
    return push!(hwtlas, Raycore.Instance(geometry, transforms; metadata=material_idx); arealight_indices)
end

function Base.push!(hwtlas::HWTLAS, geometry)
    return push!(hwtlas, Raycore.Instance(geometry))
end

"""
    push!(hwtlas::HWTLAS, mesh::GeometryBasics.Mesh, transform::Mat4f=Mat4f(I))

Add a GeometryBasics.Mesh to the HWTLAS. Per-face metadata is read from the mesh's
`face_meta` attribute (if present). Mirrors `push!(::TLAS, ::Mesh, ::Mat4f)`.
"""
function Base.push!(hwtlas::HWTLAS, mesh::GeometryBasics.Mesh, transform::Mat4f=Mat4f(I))
    nmesh = GeometryBasics.expand_faceviews(mesh)
    fs = decompose(TriangleFace{UInt32}, nmesh)
    verts = decompose(Point3f, nmesh)
    norms = Raycore.Normal3f.(decompose_normals(nmesh))
    uvs_raw = GeometryBasics.decompose_uv(nmesh)
    uvs = isnothing(uvs_raw) ? Point2f[] : Point2f.(uvs_raw)
    indices = collect(reinterpret(UInt32, fs))

    has_meta = hasproperty(nmesh, :face_meta)
    n_faces = length(fs)

    cpu_triangles = [begin
            meta = has_meta ? nmesh.face_meta[indices[3*(i-1)+1]] : UInt32(i)
            Raycore.build_triangle(verts, norms, uvs, indices, i, meta)
        end
        for i in 1:n_faces
        if !Raycore.is_degenerate_face(verts, indices, i)
    ]
    isempty(cpu_triangles) && error("Geometry has no valid triangles")

    # Extract vertex positions for Vulkan BLAS build
    n_tris = length(cpu_triangles)
    blas_vertices = Vector{NTuple{3,Float32}}(undef, n_tris * 3)
    for i in 1:n_tris
        vs = cpu_triangles[i].vertices
        for j in 1:3
            v = vs[j]
            blas_vertices[(i-1)*3 + j] = (Float32(v[1]), Float32(v[2]), Float32(v[3]))
        end
    end
    blas_indices = Vector{UInt32}(undef, n_tris * 3)
    for i in 0:(n_tris*3 - 1)
        blas_indices[i+1] = UInt32(i)
    end

    hw_blas = build_blas(blas_vertices, blas_indices)
    push!(hwtlas.blas_list, hw_blas)
    push!(hwtlas.blas_triangles, cpu_triangles)
    blas_idx = length(hwtlas.blas_list)

    offset = isempty(hwtlas.blas_offsets) ? UInt32(0) :
             hwtlas.blas_offsets[end] + UInt32(length(hwtlas.blas_triangles[end-1]))
    push!(hwtlas.blas_offsets, offset)

    for tri in cpu_triangles
        for v in tri.vertices
            hwtlas.root_aabb = union(hwtlas.root_aabb, Bounds3(Point3f(v)))
        end
    end

    start_idx = length(hwtlas.instance_blas_indices) + 1
    push!(hwtlas.instance_blas_indices, blas_idx)
    push!(hwtlas.instance_transforms, _mat4_to_vk_transform(transform))
    push!(hwtlas.instance_custom_indices, UInt32(blas_idx - 1))
    end_idx = length(hwtlas.instance_blas_indices)

    handle = TLASHandle(hwtlas.next_handle_id)
    hwtlas.next_handle_id += UInt32(1)
    hwtlas.handle_to_range[handle] = start_idx:end_idx

    hwtlas.dirty = true
    return handle
end

function Base.push!(hwtlas::HWTLAS, inst::Raycore.Instance; arealight_indices=nothing)
    # Decompose mesh → triangles (same as Raycore)
    mesh = Raycore.to_triangle_mesh(inst.geometry)
    has_per_face_mat = !isempty(mesh.material_indices)
    n_faces = div(length(mesh.indices), 3)

    cpu_triangles = [begin
            mat_key = has_per_face_mat ? mesh.material_indices[i] : inst.metadata[1]
            al_idx = !isnothing(arealight_indices) ? arealight_indices[i] : UInt32(0)
            Triangle(mesh, i, TriangleMeta(mat_key, UInt32(i), al_idx))
        end
        for i in 1:n_faces
        if !Raycore.is_degenerate(Raycore.get_vertices(mesh, i))]
    isempty(cpu_triangles) && error("Geometry has no valid triangles")

    # Extract vertices + indices for Vulkan BLAS build
    n_tris = length(cpu_triangles)
    vertices = Vector{NTuple{3,Float32}}(undef, n_tris * 3)
    for i in 1:n_tris
        verts = cpu_triangles[i].vertices
        for j in 1:3
            v = verts[j]
            vertices[(i-1)*3 + j] = (Float32(v[1]), Float32(v[2]), Float32(v[3]))
        end
    end
    indices = Vector{UInt32}(undef, n_tris * 3)
    for i in 0:(n_tris*3 - 1)
        indices[i+1] = UInt32(i)
    end

    # Build Vulkan BLAS immediately
    hw_blas = build_blas(vertices, indices)
    push!(hwtlas.blas_list, hw_blas)
    push!(hwtlas.blas_triangles, cpu_triangles)
    blas_idx = length(hwtlas.blas_list)

    # Compute offset into flat triangle array
    offset = isempty(hwtlas.blas_offsets) ? UInt32(0) :
             hwtlas.blas_offsets[end] + UInt32(length(hwtlas.blas_triangles[end-1]))
    push!(hwtlas.blas_offsets, offset)

    # Update bounding box from triangle vertices
    for tri in cpu_triangles
        for v in tri.vertices
            hwtlas.root_aabb = union(hwtlas.root_aabb, Bounds3(Point3f(v)))
        end
    end

    # Add instances (one per transform)
    start_idx = length(hwtlas.instance_blas_indices) + 1
    for transform in inst.transforms
        push!(hwtlas.instance_blas_indices, blas_idx)
        push!(hwtlas.instance_transforms, _mat4_to_vk_transform(transform))
        push!(hwtlas.instance_custom_indices, UInt32(blas_idx - 1))  # 0-based for HW RT
    end
    end_idx = length(hwtlas.instance_blas_indices)

    # Create handle
    handle = TLASHandle(hwtlas.next_handle_id)
    hwtlas.next_handle_id += UInt32(1)
    hwtlas.handle_to_range[handle] = start_idx:end_idx

    hwtlas.dirty = true
    return handle
end

# ── delete! ──

function Base.delete!(hwtlas::HWTLAS, handle::TLASHandle)::Bool
    haskey(hwtlas.handle_to_range, handle) || return false
    handle in hwtlas.deleted_handles && return false
    push!(hwtlas.deleted_handles, handle)
    hwtlas.dirty = true
    return true
end

# ── sync! — build Vulkan TLAS from accumulated BLASes ──

function Raycore.sync!(hwtlas::HWTLAS)
    hwtlas.dirty || return hwtlas

    n_instances = length(hwtlas.instance_blas_indices)
    if n_instances == 0
        hwtlas.hw_tlas = nothing
        hwtlas.hw_accel = nothing
        hwtlas.dirty = false
        return hwtlas
    end

    # Build BLAS reference list + transforms for TLAS
    hw_blas_refs = [hwtlas.blas_list[hwtlas.instance_blas_indices[i]] for i in 1:n_instances]

    hwtlas.hw_tlas = build_tlas(hw_blas_refs;
        transforms=hwtlas.instance_transforms,
        custom_indices=hwtlas.instance_custom_indices)

    # Build the HardwareAccel wrapper (holds RT pipeline + TLAS)
    # Collect all triangles into a typed vector (blas_triangles stores Vector{Any})
    all_tris_any = reduce(vcat, hwtlas.blas_triangles)
    T = typeof(all_tris_any[1])
    all_tris = T[t for t in all_tris_any]
    hwtlas.hw_accel = Lava.HardwareAccel(hwtlas.hw_tlas, all_tris, hwtlas.blas_offsets)

    # Upload triangle data + offsets to GPU
    hwtlas.tri_gpu = LavaArray(all_tris)
    hwtlas.off_gpu = LavaArray(hwtlas.blas_offsets)

    hwtlas.dirty = false
    return hwtlas
end

# ── Scene sync! for HWTLAS scenes ──

function sync!(scene::Scene{<:HWTLAS})
    Raycore.sync!(scene.accel)
    bound = Raycore.world_bound(scene.accel)
    scene.bounds[] = (bound, bounding_sphere(bound))
    return scene
end

# ── Accessors ──

Raycore.world_bound(hwtlas::HWTLAS) = hwtlas.root_aabb
Raycore.n_geometries(hwtlas::HWTLAS) = length(hwtlas.blas_list)
Raycore.n_instances(hwtlas::HWTLAS) = length(hwtlas.instance_blas_indices)
Raycore.refit_tlas!(hwtlas::HWTLAS) = nothing  # HW TLAS doesn't need refit (rebuilt on sync!)

# RayMakie accesses tlas.instances directly for `isempty(tlas.instances)`
# Provide a lightweight view
struct _HWTLASInstances
    n::Int
end
Base.isempty(x::_HWTLASInstances) = x.n == 0
Base.length(x::_HWTLASInstances) = x.n

function Base.getproperty(hwtlas::HWTLAS, s::Symbol)
    if s === :instances
        return _HWTLASInstances(length(getfield(hwtlas, :instance_blas_indices)))
    else
        return getfield(hwtlas, s)
    end
end

# ── Indirect dispatch for WorkQueue foreach (no CPU readback) ──
_gpu_ndrange(::LavaBackend, size_buf::LavaArray) = size_buf

# ── HWAdaptedAccel — GPU-adapted form of HWTLAS ──

"""
    HWAdaptedAccel

The GPU-adapted form of HWTLAS. Returned by `Adapt.adapt_structure(to, ::HWTLAS)`.
Dispatches `vp_trace_rays!` and `vp_trace_shadow_rays!` to hardware implementations.
"""
struct HWAdaptedAccel
    hwtlas::HWTLAS
end

function Adapt.adapt_structure(to, hwtlas::HWTLAS)
    HWAdaptedAccel(hwtlas)
end

# ── fill_aux_buffers! override for HWTLAS scenes ──
# The default kernel calls intersect! inside compute, which can't use HW RT.
# Fill with defaults (depth=miss, normal=0, albedo=0.8) — rendering is correct,
# denoising won't have useful normals/depth but still functions.
function fill_aux_buffers!(film::Film, scene::Scene{<:HWAdaptedAccel}, camera; has_infinite_lights::Bool=false)
    fill!(film.depth, 0f0)  # Must be finite — Inf triggers escaped-pixel masking in postprocess
    fill!(film.normal, Vec3f(0f0, 0f0, 0f0))
    fill!(film.albedo, RGB{Float32}(0.8f0, 0.8f0, 0.8f0))
    return nothing
end

# ── PrecomputedHitsAccel ──
# A "fake" acceleration structure that returns pre-computed RT results.
# The kernel calls Raycore.closest_hit(precomputed, ray), which reads from
# the results buffer using the thread's GlobalInvocationID as index.

struct PrecomputedHitsAccel{R, T, O}
    results::R       # RTHitResult array (1 per ray)
    triangles::T     # Triangle{TMetadata} array (all primitives, flat)
    offsets::O       # UInt32 array (per-BLAS offset into triangles)
end

function Adapt.adapt_structure(to, p::PrecomputedHitsAccel)
    PrecomputedHitsAccel(
        Adapt.adapt(to, p.results),
        Adapt.adapt(to, p.triangles),
        Adapt.adapt(to, p.offsets),
    )
end

@propagate_inbounds function Raycore.closest_hit(accel::PrecomputedHitsAccel, ray)
    tid = lava_global_invocation_id_x() + UInt32(1)  # 1-based
    result = accel.results[tid]

    if result.hit == UInt32(0)
        dummy = accel.triangles[1]
        return (false, dummy, 0f0, SVector{3,Float32}(1f0, 0f0, 0f0))
    end

    tri_idx = Int(accel.offsets[result.instance_custom_index + UInt32(1)]) +
              Int(result.primitive_id) + 1
    tri = accel.triangles[tri_idx]

    w = 1f0 - result.bary_u - result.bary_v
    bary = SVector{3,Float32}(w, result.bary_u, result.bary_v)

    return (true, tri, result.t, bary)
end

# ── Ray Extraction Kernel ──

@kernel function _extract_rays_kernel!(ray_buf, @Const(queue_items), @Const(queue_size))
    i = @index(Global)
    if i <= queue_size[1]
        work = queue_items[i]
        ray = work.ray
        ray_buf[i] = RTRay(
            Float32(ray.o[1]), Float32(ray.o[2]), Float32(ray.o[3]),
            0.001f0,
            Float32(ray.d[1]), Float32(ray.d[2]), Float32(ray.d[3]),
            ray.t_max,
        )
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Dispatch overrides — route to HW RT when accel is HWAdaptedAccel
# ══════════════════════════════════════════════════════════════════════════════

# Camera medium detection: use bounding box check (no CPU BVH needed)
# For now, assume camera is outside all media (correct for most scenes).
# A proper implementation would do a single HW RT trace on GPU.
function _detect_initial_medium(backend, accel::HWAdaptedAccel, mi, pos, vp::VolPath)
    return Raycore.SetKey()  # No medium (camera outside all geometry)
end

# Primary ray tracing: HW RT dispatch (indirect — zero CPU readbacks)
function vp_trace_rays!(state::VolPathState, accel::HWAdaptedAccel, media_interfaces, materials, ::VolPath)
    hwtlas = accel.hwtlas
    hw = hwtlas.hw_accel

    input_queue = state.current_ray_queue == :a ? state.ray_queue_a : state.ray_queue_b
    backend = KernelAbstractions.get_backend(input_queue.items)

    # Use queue capacity for buffer sizing (no CPU readback for actual size)
    cap = Int(input_queue.capacity)
    if hwtlas.primary_ray_buf === nothing || length(hwtlas.primary_ray_buf) < cap
        hwtlas.primary_ray_buf = LavaArray{RTRay}(undef, cap)
        hwtlas.primary_result_buf = LavaArray{RTHitResult}(undef, cap)
    end
    ray_buf = hwtlas.primary_ray_buf
    result_buf = hwtlas.primary_result_buf

    # Phase 1: Extract rays (indirect dispatch — reads n_rays from queue.size on GPU)
    extract_kernel! = _extract_rays_kernel!(backend, 256)
    extract_kernel!(ray_buf, input_queue.items, input_queue.size; ndrange=input_queue.size)
    # No vk_flush!() — stays in same command buffer

    # Phase 2: RT dispatch — indirect (reads n_rays from queue.size on GPU)
    trace_closest_hits_indirect!(result_buf, ray_buf, hw, input_queue.size)

    # Phase 3: Create precomputed accel with triangle data on GPU
    precomputed = PrecomputedHitsAccel(result_buf, hwtlas.tri_gpu, hwtlas.off_gpu)

    # Phase 4: Run original trace kernel (indirect dispatch via queue.size)
    foreach(vp_trace_rays_kernel!,
        input_queue,
        state.medium_sample_queue,
        state.escaped_queue,
        state.hit_surface_queue,
        precomputed,
        media_interfaces,
        materials,
    )
    return nothing
end

# ══════════════════════════════════════════════════════════════════════════════
# Hardware-Accelerated Shadow Ray Tracing (Multi-Round Dispatch)
# ══════════════════════════════════════════════════════════════════════════════

const _SS4 = SampledSpectrum{4}
const _SW4 = SampledWavelengths{4}
const _SK = Raycore.SetKey

"""
    ShadowIterState

Per-shadow-ray state for iterative hardware-accelerated shadow tracing.
"""
struct ShadowIterState
    Ld::_SS4
    r_u_path::_SS4
    r_l_path::_SS4
    lambda::_SW4
    pixel_index::Int32
    ray_o::Point3f
    dir::Vec3f
    t_remaining::Float32
    medium_idx::_SK
    T_ray::_SS4
    tr_r_u::_SS4
    tr_r_l::_SS4
    active::UInt32
    visible::UInt32
end

@kernel inbounds=true function _init_shadow_states_kernel!(
    states, @Const(queue_items), @Const(queue_size), @Const(n::Int32)
)
    i = @index(Global)
    if i <= n
        work = queue_items[i]
        states[i] = ShadowIterState(
            work.Ld, work.r_u, work.r_l, work.lambda, work.pixel_index,
            Point3f(work.ray.o), Vec3f(work.ray.d), work.t_max, work.medium_idx,
            _SS4(1f0), _SS4(1f0), _SS4(1f0),
            UInt32(1), UInt32(1),
        )
    end
end

@kernel inbounds=true function _extract_shadow_rays2_kernel!(
    ray_buf, @Const(states), @Const(n::Int32)
)
    i = @index(Global)
    if i <= n
        st = states[i]
        if st.active == UInt32(1) && st.t_remaining >= 1f-6
            ray_buf[i] = RTRay(
                Float32(st.ray_o[1]), Float32(st.ray_o[2]), Float32(st.ray_o[3]),
                0.001f0,
                Float32(st.dir[1]), Float32(st.dir[2]), Float32(st.dir[3]),
                st.t_remaining,
            )
        else
            ray_buf[i] = RTRay(0f0, 0f0, 0f0, 0f0, 0f0, 0f0, 1f0, -1f0)
        end
    end
end

@kernel inbounds=true function _process_shadow_round_kernel!(
    states, @Const(result_buf), @Const(tri_gpu), @Const(off_gpu),
    media_interfaces, media, materials, rgb2spec_table,
    @Const(n::Int32)
)
    i = @index(Global)
    if i <= n
        st = states[i]
        if st.active == UInt32(1)
            if st.t_remaining < 1f-6
                states[i] = ShadowIterState(
                    st.Ld, st.r_u_path, st.r_l_path, st.lambda, st.pixel_index,
                    st.ray_o, st.dir, st.t_remaining, st.medium_idx,
                    st.T_ray, st.tr_r_u, st.tr_r_l,
                    UInt32(0), UInt32(1))
            else
                result = result_buf[i]
                if result.hit == UInt32(0)
                    T_new = st.T_ray
                    r_u_new = st.tr_r_u
                    r_l_new = st.tr_r_l
                    if has_medium(st.medium_idx)
                        seg_T, seg_r_u, seg_r_l = compute_transmittance_ratio_tracking(
                            rgb2spec_table, media, st.medium_idx,
                            st.ray_o, st.dir, st.t_remaining, st.lambda)
                        T_new = T_new * seg_T
                        r_u_new = r_u_new * seg_r_u
                        r_l_new = r_l_new * seg_r_l
                    end
                    states[i] = ShadowIterState(
                        st.Ld, st.r_u_path, st.r_l_path, st.lambda, st.pixel_index,
                        st.ray_o, st.dir, st.t_remaining, st.medium_idx,
                        T_new, r_u_new, r_l_new,
                        UInt32(0), UInt32(1))
                else
                    tri_idx = Int(off_gpu[result.instance_custom_index + UInt32(1)]) +
                              Int(result.primitive_id) + 1
                    tri = tri_gpu[tri_idx]
                    t_hit = result.t
                    w_bary = 1f0 - result.bary_u - result.bary_v
                    bary = SVector{3,Float32}(w_bary, result.bary_u, result.bary_v)

                    mi_idx = tri.metadata.medium_interface_idx
                    mi = media_interfaces[mi_idx]
                    ng = vp_compute_geometric_normal(tri)
                    entering = dot(Vec3f(st.dir), ng) < 0f0

                    is_xmit = is_medium_transition(mi)
                    handled = false

                    if is_xmit
                        T_new = st.T_ray
                        r_u_new = st.tr_r_u
                        r_l_new = st.tr_r_l
                        if has_medium(st.medium_idx)
                            seg_T, seg_r_u, seg_r_l = compute_transmittance_ratio_tracking(
                                rgb2spec_table, media, st.medium_idx,
                                st.ray_o, st.dir, t_hit, st.lambda)
                            T_new = T_new * seg_T
                            r_u_new = r_u_new * seg_r_u
                            r_l_new = r_l_new * seg_r_l
                        end
                        if is_black(T_new)
                            states[i] = ShadowIterState(
                                st.Ld, st.r_u_path, st.r_l_path, st.lambda, st.pixel_index,
                                st.ray_o, st.dir, st.t_remaining, st.medium_idx,
                                T_new, r_u_new, r_l_new,
                                UInt32(0), UInt32(1))
                        else
                            new_medium = get_crossing_medium(mi, entering)
                            new_ray_o = Point3f(st.ray_o + st.dir * (t_hit + 1f-4))
                            new_t_rem = st.t_remaining - t_hit - 1f-4
                            states[i] = ShadowIterState(
                                st.Ld, st.r_u_path, st.r_l_path, st.lambda, st.pixel_index,
                                new_ray_o, st.dir, new_t_rem, new_medium,
                                T_new, r_u_new, r_l_new,
                                UInt32(1), UInt32(1))
                        end
                        handled = true
                    end

                    if !is_xmit && !handled
                        uv = vp_compute_uv_barycentric(tri, bary)
                        mat_idx = mi.material
                        alpha = get_surface_alpha_dispatch(materials, mat_idx, uv)
                        alpha_pass = false

                        if alpha < 1f0
                            rng = pcg32_init(pbrt_hash(st.ray_o), pbrt_hash(st.dir))
                            alpha_u, _ = pcg32_uniform_f32(rng)
                            if alpha_u > alpha
                                T_new = st.T_ray
                                r_u_new = st.tr_r_u
                                r_l_new = st.tr_r_l
                                if has_medium(st.medium_idx)
                                    seg_T, seg_r_u, seg_r_l = compute_transmittance_ratio_tracking(
                                        rgb2spec_table, media, st.medium_idx,
                                        st.ray_o, st.dir, t_hit, st.lambda)
                                    T_new = T_new * seg_T
                                    r_u_new = r_u_new * seg_r_u
                                    r_l_new = r_l_new * seg_r_l
                                end
                                new_ray_o = Point3f(st.ray_o + st.dir * (t_hit + 1f-4))
                                new_t_rem = st.t_remaining - t_hit - 1f-4
                                states[i] = ShadowIterState(
                                    st.Ld, st.r_u_path, st.r_l_path, st.lambda, st.pixel_index,
                                    new_ray_o, st.dir, new_t_rem, st.medium_idx,
                                    T_new, r_u_new, r_l_new,
                                    UInt32(1), UInt32(1))
                                alpha_pass = true
                            end
                        end

                        if !alpha_pass
                            states[i] = ShadowIterState(
                                st.Ld, st.r_u_path, st.r_l_path, st.lambda, st.pixel_index,
                                st.ray_o, st.dir, st.t_remaining, st.medium_idx,
                                _SS4(0f0), _SS4(1f0), _SS4(1f0),
                                UInt32(0), UInt32(0))
                        end
                    end
                end
            end
        end
    end
end

@kernel inbounds=true function _count_active_shadows_kernel!(
    counter, @Const(states), @Const(n::Int32)
)
    i = @index(Global)
    if i <= n
        if states[i].active == UInt32(1)
            Atomix.@atomic counter[1] += Int32(1)
        end
    end
end

@kernel inbounds=true function _finalize_shadow_kernel!(
    states, pixel_L, @Const(n::Int32)
)
    i = @index(Global)
    if i <= n
        st = states[i]
        if st.visible == UInt32(1) && !is_black(st.T_ray)
            mis_weight = st.r_u_path * st.tr_r_u + st.r_l_path * st.tr_r_l
            mis_denom = average(mis_weight)
            if mis_denom > 1f-10
                final_L = st.Ld * st.T_ray / mis_denom
                if !is_black(final_L)
                    pixel_idx = st.pixel_index
                    base_idx = (pixel_idx - Int32(1)) * Int32(4)
                    accumulate_spectrum!(pixel_L, base_idx, final_L)
                end
            end
        end
    end
end

# Shadow ray tracing dispatch (indirect — minimal CPU readbacks)
function vp_trace_shadow_rays!(state::VolPathState, accel::HWAdaptedAccel, media_interfaces, media, materials, ::VolPath)
    hwtlas = accel.hwtlas
    hw = hwtlas.hw_accel

    shadow_queue = state.shadow_queue
    backend = KernelAbstractions.get_backend(shadow_queue.items)

    # Use queue capacity for buffer sizing (no CPU readback)
    cap = Int(shadow_queue.capacity)
    if hwtlas.shadow_states === nothing || length(hwtlas.shadow_states) < cap
        hwtlas.shadow_states = LavaArray{ShadowIterState}(undef, cap)
        hwtlas.shadow_ray_buf = LavaArray{RTRay}(undef, cap)
        hwtlas.shadow_result_buf = LavaArray{RTHitResult}(undef, cap)
        hwtlas.shadow_active_counter = LavaArray{Int32}(undef, 1)
    end
    states = hwtlas.shadow_states
    ray_buf = hwtlas.shadow_ray_buf
    result_buf = hwtlas.shadow_result_buf
    active_counter = hwtlas.shadow_active_counter
    n_rays_gpu = shadow_queue.size  # GPU-resident ray count

    # Initialize shadow ray iteration state (indirect — reads count from GPU)
    init_k! = _init_shadow_states_kernel!(backend, 256)
    init_k!(states, shadow_queue.items, shadow_queue.size, Int32(cap); ndrange=n_rays_gpu)
    # No flush — stays in command buffer

    extract_k! = _extract_shadow_rays2_kernel!(backend, 256)
    process_k! = _process_shadow_round_kernel!(backend, 256)
    count_k! = _count_active_shadows_kernel!(backend, 256)

    # Round 1: full dispatch (all rays active initially)
    extract_k!(ray_buf, states, Int32(cap); ndrange=n_rays_gpu)
    trace_closest_hits_indirect!(result_buf, ray_buf, hw, n_rays_gpu)
    process_k!(states, result_buf, hwtlas.tri_gpu, hwtlas.off_gpu,
               media_interfaces, media, materials, state.rgb2spec_table,
               Int32(cap); ndrange=n_rays_gpu)

    # Rounds 2-10: RT trace uses active counter (0 active = hardware no-op)
    for _round in 2:10
        fill!(active_counter, Int32(0))
        count_k!(active_counter, states, Int32(cap); ndrange=n_rays_gpu)

        extract_k!(ray_buf, states, Int32(cap); ndrange=n_rays_gpu)
        trace_closest_hits_indirect!(result_buf, ray_buf, hw, active_counter)
        process_k!(states, result_buf, hwtlas.tri_gpu, hwtlas.off_gpu,
                   media_interfaces, media, materials, state.rgb2spec_table,
                   Int32(cap); ndrange=n_rays_gpu)
    end

    # Finalize — accumulate completed visible rays to pixel_L
    finalize_k! = _finalize_shadow_kernel!(backend, 256)
    finalize_k!(states, state.pixel_L, Int32(cap); ndrange=n_rays_gpu)

    return nothing
end

# ══════════════════════════════════════════════════════════════════════════════
# Any-Hit Shadow Shader — Handles Medium Transitions in Hardware
# ══════════════════════════════════════════════════════════════════════════════
#
# When tracing shadow rays with an any-hit shader, the RT hardware calls
# the any-hit function at every potential intersection. If the surface is a
# medium transition (glass boundary), we call OpIgnoreIntersectionKHR to
# continue traversal. Opaque surfaces let the hit be accepted normally.
#
# This eliminates the multi-round extract→trace→process loop for scenes
# with transparent surfaces — a single RT dispatch handles all transparency.

"""
Raygen shader for shadow RT with any-hit support.
Same as `_hw_raygen` but receives extra args for the any-hit shader's BDA access:
  (rays, results, tri_gpu, off_gpu, media_interfaces)
The extra args aren't used by raygen directly — they're in the shared BDA buffer
so the any-hit shader can load them via the same push constant.
"""
function _hw_raygen_shadow(rays::Ptr{RTRay}, results::Ptr{RTHitResult},
                           tri_gpu, off_gpu, media_interfaces)
    lid = lava_rt_launch_id_x()

    ray = unsafe_load(rays, lid + 1)

    _lava_rt_payload_store_f32_at(0f0, UInt32(0))   # hit=0
    _lava_rt_payload_store_f32_at(-1f0, UInt32(1))   # t=-1

    _lava_rt_trace_ray(
        UInt32(0),    # flags (no special flags — any-hit handles transparency)
        UInt32(0xFF), # cull mask
        UInt32(0),    # sbt offset
        UInt32(0),    # sbt stride
        UInt32(0),    # miss index
        ray.origin_x, ray.origin_y, ray.origin_z, ray.tmin,
        ray.dir_x, ray.dir_y, ray.dir_z, ray.tmax
    )

    hit  = _lava_rt_payload_load_f32_at(UInt32(0))
    t    = _lava_rt_payload_load_f32_at(UInt32(1))
    pid  = _lava_rt_payload_load_f32_at(UInt32(2))
    ci   = _lava_rt_payload_load_f32_at(UInt32(3))
    bu   = _lava_rt_payload_load_f32_at(UInt32(4))
    bv   = _lava_rt_payload_load_f32_at(UInt32(5))

    result = RTHitResult(
        reinterpret(UInt32, hit), t,
        reinterpret(UInt32, pid), reinterpret(UInt32, ci),
        bu, bv, UInt32(0), UInt32(0)
    )
    unsafe_store!(results, result, lid + 1)
    return nothing
end

"""
Any-hit shader for shadow rays — checks medium transitions.

Called by the RT hardware at every potential intersection during traversal.
If the hit surface is a medium transition (glass/dielectric boundary),
calls OpIgnoreIntersectionKHR to skip it and continue traversal.
Opaque surfaces are accepted normally (closest-hit shader runs).

BDA arg buffer layout matches `_hw_raygen_shadow`:
  (rays, results, tri_gpu, off_gpu, media_interfaces)
"""
function _hw_anyhit_shadow(rays::Ptr{RTRay}, results::Ptr{RTHitResult},
                           tri_gpu, off_gpu, media_interfaces)
    prim_id = lava_rt_primitive_id()
    inst_idx = lava_rt_instance_custom_index()

    # Look up BLAS offset, then triangle
    tri_offset = unsafe_load(off_gpu, inst_idx + UInt32(1))
    tri = unsafe_load(tri_gpu, Int(tri_offset) + Int(prim_id) + 1)

    # Check medium interface — transparent surfaces should be skipped
    mi_idx = tri.metadata.medium_interface_idx
    mi = unsafe_load(media_interfaces, Int(mi_idx))

    # is_medium_transition: inside != outside (different media on each side)
    if mi.inside.type_idx != mi.outside.type_idx || mi.inside.vec_idx != mi.outside.vec_idx
        _lava_rt_ignore_intersection()
    end
    # Opaque: do nothing — hardware accepts the hit, closest-hit shader stores result
    return nothing
end

"""
    _ensure_anyhit_pipeline!(hwtlas::HWTLAS)

Lazily create the any-hit RT pipeline for shadow rays.
Uses `_hw_raygen_shadow` + `_hw_anyhit_shadow` with the concrete
Triangle and MediumInterfaceIdx types from the scene's triangle data.
"""
function _ensure_anyhit_pipeline!(hwtlas::HWTLAS)
    hw = hwtlas.hw_accel
    hw === nothing && error("HWTLAS not synced — call sync! first")
    hw.anyhit_pipeline !== nothing && return hw

    set_anyhit_pipeline!(hw, _hw_anyhit_shadow, _hw_raygen_shadow)
    return hw
end
