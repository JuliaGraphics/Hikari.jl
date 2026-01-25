module Hikari

using Base: @propagate_inbounds
import FileIO
using ImageCore
using ImageIO
using GeometryBasics
using LinearAlgebra
using StaticArrays
using ProgressMeter
using StructArrays
using Atomix
using KernelAbstractions
using Raycore
using Zlib_jll
using Adapt
using KernelAbstractions: @kernel, @index, @Const
import KernelAbstractions as KA

# Re-export Raycore types and functions that Trace uses
import Raycore: AbstractRay, Ray, RayDifferentials, apply, check_direction, scale_differentials
import Raycore: Bounds2, Bounds3, area, surface_area, diagonal, maximum_extent, offset, is_valid, inclusive_sides, expand
import Raycore: distance, distance_squared, bounding_sphere
# Note: lerp is defined in spectrum.jl for Spectrum, Float32, and Point3f
import Raycore: Transformation, translate, scale, rotate, rotate_x, rotate_y, rotate_z, look_at, perspective
import Raycore: swaps_handedness, has_scale
import Raycore: AbstractShape, Triangle, TriangleMesh
import Raycore: AccelPrimitive, BVH, TLAS, Instance, InstanceHandle, world_bound, closest_hit, any_hit
import Raycore: Normal3f, intersect, intersect_p
import Raycore: is_dir_negative, increase_hit, intersect_p!
import Raycore: to_gpu
import Raycore: sum_unrolled, reduce_unrolled, for_unrolled, map_unrolled, getindex_unrolled
import Raycore: HeteroVecIndex, MultiTypeVec, StaticMultiTypeVec, with_index, is_invalid, n_slots

abstract type Spectrum end
abstract type Light end
abstract type Material end
abstract type BxDF end
abstract type Integrator end

# Default no-op close for integrators without cached state
Base.close(::Integrator) = nothing

const Radiance = UInt8(1)
const Importance = UInt8(2)

struct Reflect end
struct Transmit end

const DO_ASSERTS = false
macro real_assert(expr, msg="")
    if DO_ASSERTS
        esc(:(@assert $expr $msg))
    else
        return :()
    end
end

# GeometryBasics.@fixed_vector Normal StaticVector
# Normal3f is now imported from Raycore
include("mempool.jl")


const Maybe{T} = Union{T,Nothing}

function get_progress_bar(n::Integer, desc::String = "Progress")
    Progress(
        n, desc = desc, dt = 1,
        barglyphs = BarGlyphs("[=> ]"), barlen = 50, color = :white,
    )
end

@propagate_inbounds maybe_copy(v::Maybe)::Maybe  = v isa Nothing ? v : copy(v)

@propagate_inbounds function concentric_sample_disk(u::Point2f)::Point2f
    # Map uniform random numbers to [-1, 1].
    offset_x = 2f0 * u[1] - 1f0
    offset_y = 2f0 * u[2] - 1f0

    # Compute r and θ - avoid zero check, just compute through
    # (The zero case is extremely rare and the math will naturally produce ~0)
    abs_x = abs(offset_x)
    abs_y = abs(offset_y)

    # Add tiny epsilon to avoid division by zero without branching
    safe_offset_x = offset_x + 1.0f-10
    safe_offset_y = offset_y + 1.0f-10

    is_x_larger = abs_x > abs_y
    r = ifelse(is_x_larger, offset_x, offset_y)
    θ = ifelse(is_x_larger,
               (offset_y / safe_offset_x) * π / 4f0,
               π / 2f0 - (offset_x / safe_offset_y) * π / 4f0)

    # Direct computation and return - no conditional selection
    return Point2f(r * cos(θ), r * sin(θ))
end

function cosine_sample_hemisphere(u::Point2f)::Vec3f
    d = concentric_sample_disk(u)
    z = √max(0f0, 1f0 - d[1]^2 - d[2]^2)
    Vec3f(d[1], d[2], z)
end

function uniform_sample_sphere(u::Point2f)::Vec3f
    z = 1f0 - 2f0 * u[1]
    r = √(max(0f0, 1f0 - z^2))
    ϕ = 2f0 * π * u[2]
    Vec3f(r * cos(ϕ), r * sin(ϕ), z)
end

function uniform_sample_cone(u::Point2f, cosθ_max::Float32)::Vec3f
    cosθ = 1f0 - u[1] + u[1] * cosθ_max
    sinθ = √(1f0 - cosθ^2)
    ϕ = u[2] * 2f0 * π
    Vec3f(cos(ϕ) * sinθ, sin(ϕ) * sinθ, cosθ)
end

function uniform_sample_cone(
    u::Point2f, cosθ_max::Float32, x::Vec3f, y::Vec3f, z::Vec3f,
)::Vec3f
    cosθ = 1f0 - u[1] + u[1] * cosθ_max
    sinθ = √(1f0 - cosθ^2)
    ϕ = u[2] * 2f0 * π
    x * cos(ϕ) * sinθ + y * sin(ϕ) * sinθ + z * cosθ
end

@propagate_inbounds uniform_sphere_pdf()::Float32  = 1f0 / (4f0 * π)

@propagate_inbounds function uniform_cone_pdf(cosθ_max::Float32)::Float32
    1f0 / (2f0 * π * (1f0 - cosθ_max))
end

sum_mul(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

"""
The shading coordinate system gives a frame for expressing directions
in spherical coordinates (θ, ϕ).
The angle θ is measured from the given direction to the z-axis
and ϕ is the angle formed with the x-axis after projection
of the direction onto xy-plane.

Since normal is `(0, 0, 1) → cos_θ = n · w = (0, 0, 1) ⋅ w = w.z`.
"""
@propagate_inbounds cos_θ(w::Vec3f) = w[3]
@propagate_inbounds sin_θ2(w::Vec3f) = max(0f0, 1f0 - cos_θ(w) * cos_θ(w))
@propagate_inbounds sin_θ(w::Vec3f) = √(sin_θ2(w))
@propagate_inbounds tan_θ(w::Vec3f) = sin_θ(w) / cos_θ(w)

@propagate_inbounds function cos_ϕ(w::Vec3f)
    sinθ = sin_θ(w)
    sinθ ≈ 0f0 ? 1f0 : clamp(w[1] / sinθ, -1f0, 1f0)
end
@propagate_inbounds function sin_ϕ(w::Vec3f)
    sinθ = sin_θ(w)
    sinθ ≈ 0f0 ? 1f0 : clamp(w[2] / sinθ, -1f0, 1f0)
end

# reflect is defined in materials/spectral-eval.jl

function partition!(x::Vector, range::UnitRange, predicate::Function)
    left = range[1]
    for i in range
        if left != i && predicate(x[i])
            x[i], x[left] = x[left], x[i]
            left += 1
        end
    end
    left
end

# coordinate_system is defined in materials/spectral-eval.jl

function spherical_direction(sin_θ::Float32, cos_θ::Float32, ϕ::Float32)
    Vec3f(sin_θ * cos(ϕ), sin_θ * sin(ϕ), cos_θ)
end
function spherical_direction(
    sin_θ::Float32, cos_θ::Float32, ϕ::Float32,
    x::Vec3f, y::Vec3f, z::Vec3f,
)
    sin_θ * cos(ϕ) * x + sin_θ * sin(ϕ) * y + cos_θ * z
end

spherical_θ(v::Vec3f) = acos(clamp(v[3], -1f0, 1f0))
function spherical_ϕ(v::Vec3f)
    p = atan(v[2], v[1])
    p < 0 ? p + 2f0 * π : p
end


"""
Flip normal `n` so that it lies in the same hemisphere as `v`.
"""
@propagate_inbounds face_forward(n, v) = (n ⋅ v) < 0 ? -n : n

include("spectrum.jl")
include("random.jl")
include("surface_interaction.jl")

# Abstract scene type for dispatch - allows both mutable and immutable implementations
abstract type AbstractScene end

# Scene stores lights, accelerator, materials, and media directly.
# Materials and media use MultiTypeVec (or StaticMultiTypeVec for GPU/kernels).
# Textures are stored within the materials MultiTypeVec (materials have TextureRef fields).
mutable struct Scene{Accel, L<:NTuple{N, Light} where N, MatVec<:AbstractVector, MedVec<:AbstractVector} <: AbstractScene
    lights::L
    accel::Accel  # BVH, TLAS, or any accelerator supporting closest_hit/any_hit
    materials::MatVec  # MultiTypeVec or StaticMultiTypeVec
    media::MedVec  # MultiTypeVec or StaticMultiTypeVec
    bound::Bounds3
    world_center::Point3f
    world_radius::Float32
end

# Immutable scene for e.g. GPU use - required for GPU kernels (must be bitstype)
struct ImmutableScene{Accel, L<:NTuple{N, Light} where N, MatVec<:AbstractVector, MedVec<:AbstractVector} <: AbstractScene
    lights::L
    accel::Accel
    materials::MatVec
    media::MedVec
    bound::Bounds3
    world_center::Point3f
    world_radius::Float32
end

function Scene(lights::Union{Tuple,AbstractVector}, accel, materials, media)
    ltuple = Tuple(lights)
    bounds = world_bound(accel)
    world_center, world_radius = bounding_sphere(bounds)
    Scene(ltuple, accel, materials, media, bounds, world_center, world_radius)
end

# Convert Scene to ImmutableScene for GPU rendering
ImmutableScene(s::Scene) = ImmutableScene(s.lights, s.accel, s.materials, s.media, s.bound, s.world_center, s.world_radius)
ImmutableScene(s::ImmutableScene) = s  # Already immutable

# Common interface for both scene types
@propagate_inbounds function intersect!(scene::AbstractScene, ray::AbstractRay)
    intersect!(scene.accel, ray)
end

@propagate_inbounds function intersect_p(scene::AbstractScene, ray::AbstractRay)
    hit_found, _, _, _ = any_hit(scene.accel, ray)
    return hit_found
end

# Pretty printing for Scene
function Base.show(io::IO, ::MIME"text/plain", scene::Scene)
    n_lights = length(scene.lights)
    n_material_types = length(scene.materials)
    n_materials_total = sum(length, scene.materials; init=0)
    accel = scene.accel
    accel_type = nameof(typeof(accel))

    println(io, "Scene ($accel_type):")
    println(io, "  Lights:     ", n_lights)
    if n_lights > 0
        for (i, light) in enumerate(scene.lights)
            println(io, "    [", i, "] ", typeof(light).name.name)
        end
    end
    println(io, "  Materials:  ", n_materials_total, " ($n_material_types types)")
    if accel isa TLAS
        n_instances = length(accel.instances)
        n_geometries = length(accel.blas_array)
        println(io, "  Geometries: ", n_geometries)
        println(io, "  Instances:  ", n_instances)
    elseif accel isa BVH
        n_triangles = length(accel.primitives)
        println(io, "  Triangles:  ", n_triangles)
    end
    print(io,   "  Bounds:     ", scene.bound.p_min, " to ", scene.bound.p_max)
end

function Base.show(io::IO, scene::Scene)
    if get(io, :compact, false)
        n_lights = length(scene.lights)
        accel_type = nameof(typeof(scene.accel))
        n_material_types = length(scene.materials)
        print(io, "Scene($accel_type, lights=$n_lights, materials=$n_material_types)")
    else
        show(io, MIME("text/plain"), scene)
    end
end

# spawn_ray functions are now in Raycore, but we need versions for our SurfaceInteraction
# Raycore only has spawn_ray for its simpler Interaction type

"""
    MaterialIndex

Metadata type for triangles that stores material lookup information.
Now an alias for `HeteroVecIndex` from Raycore.
- `type_idx`: Which tuple slot (1-based) in the materials tuple
- `vec_idx`: Index within that material type's array
"""
const MaterialIndex = HeteroVecIndex

# ============================================================================
# with_material - delegates to Raycore.with_index for StaticMultiTypeVec
# ============================================================================

"""
    with_material(f, materials::StaticMultiTypeVec, idx::MaterialIndex, args...)

Execute function `f` with the material at index `idx`, passing additional `args`.
Delegates to `Raycore.with_index` for type-stable dispatch.
"""
@propagate_inbounds function with_material(f, materials::StaticMultiTypeVec, idx::MaterialIndex, args...)
    return with_index(f, materials, idx, args...)
end

# ============================================================================
# shade_material - delegates to material's shade method via with_index
# ============================================================================

"""
    shade_material(materials::StaticMultiTypeVec, idx::MaterialIndex, ray, si, scene, beta, depth, max_depth)

Shade a surface hit using the material at the given index.
Dispatches to the appropriate material's `shade` method via `with_index`.
"""
@propagate_inbounds function shade_material(
    materials::StaticMultiTypeVec, idx::MaterialIndex,
    ray, si, scene, beta, depth::Int32, max_depth::Int32
)
    return with_index(shade, materials, idx, ray, si, scene, beta, depth, max_depth)
end

# Calculate partial derivatives for texture mapping
function partial_derivatives(vs::AbstractVector{Point3f}, uv::AbstractVector{Point2f})
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δp_13, δp_23 = Vec3f(vs[1] - vs[3]), Vec3f(vs[2] - vs[3])
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    if det ≈ 0f0
        v = normalize((vs[3] - vs[1]) × (vs[2] - vs[1]))
        ∂p∂u, ∂p∂v = coordinate_system(Vec3f(v))
        return ∂p∂u, ∂p∂v
    end
    inv_det = 1f0 / det
    ∂p∂u = Vec3f(δuv_23[2] * δp_13 - δuv_13[2] * δp_23) * inv_det
    ∂p∂v = Vec3f(-δuv_23[1] * δp_13 + δuv_13[1] * δp_23) * inv_det
    return ∂p∂u, ∂p∂v
end

# Convert Raycore.Triangle intersection result to Trace SurfaceInteraction
@propagate_inbounds function triangle_to_surface_interaction(triangle::Triangle, ray::AbstractRay, bary_coords::StaticVector{3,Float32})::SurfaceInteraction
    # Get triangle data
    verts = Raycore.vertices(triangle)
    tex_coords = Raycore.uvs(triangle)

    pos_deriv_u, pos_deriv_v = partial_derivatives(verts, tex_coords)

    # Interpolate hit point and texture coordinates using barycentric coordinates
    hit_point = sum_mul(bary_coords, verts)
    hit_uv = sum_mul(bary_coords, tex_coords)

    # Calculate surface normal from triangle edges
    edge1 = verts[2] - verts[1]
    edge2 = verts[3] - verts[1]
    normal = normalize(edge1 × edge2)

    # Create surface interaction data at hit point
    surf_interact = SurfaceInteraction(
        normal, hit_point, ray.time, -ray.d, hit_uv,
        pos_deriv_u, pos_deriv_v, Normal3f(0f0), Normal3f(0f0)
    )

    # Initialize shading geometry from triangle normals/tangents if available
    t_normals = Raycore.normals(triangle)
    t_tangents = Raycore.tangents(triangle)

    has_normals = !all(x -> all(isnan, x), t_normals)
    has_tangents = !all(x -> all(isnan, x), t_tangents)

    if !has_normals && !has_tangents
        return surf_interact
    end

    # Initialize shading normal
    shading_normal = surf_interact.core.n
    if has_normals
        shading_normal = normalize(sum_mul(bary_coords, t_normals))
    end

    # Calculate shading tangent
    shading_tangent = Vec3f(0)
    if has_tangents
        shading_tangent = normalize(sum_mul(bary_coords, t_tangents))
    else
        shading_tangent = normalize(pos_deriv_u)
    end

    # Calculate shading bitangent
    shading_bitangent = Vec3f(shading_normal × shading_tangent)

    if (shading_bitangent ⋅ shading_bitangent) > 0f0
        shading_bitangent = Vec3f(normalize(shading_bitangent))
        shading_tangent = Vec3f(shading_bitangent × shading_normal)
    else
        _, shading_tangent, shading_bitangent = coordinate_system(Vec3f(shading_normal))
    end

    return set_shading_geometry(
        surf_interact,
        shading_tangent,
        shading_bitangent,
        Normal3f(0f0),
        Normal3f(0f0),
        true
    )
end


# Intersect TLAS - returns hit info, primitive, and SurfaceInteraction
# The primitive (triangle) contains material_type and material_idx for dispatch
@propagate_inbounds function intersect!(accel::TLAS, ray::AbstractRay)
    hit_found, triangle, distance, bary_coords, instance_id = closest_hit(accel, ray)

    if !hit_found
        return false, triangle, SurfaceInteraction()
    end

    # Convert to SurfaceInteraction (in local/BLAS space)
    interaction = triangle_to_surface_interaction(triangle, ray, bary_coords)

    # Transform surface interaction to world space using instance transform
    # instance_id is 1-based array index into accel.instances (set during TLAS construction)
    # Use it directly instead of searching - this ensures we get the current transform
    # even after updates via update_transform!
    if instance_id >= 1 && instance_id <= length(accel.instances)
        inst = accel.instances[instance_id]
        transform = inst.transform
        inv_transform = inst.inv_transform

        # Transform hit point to world space
        local_p = interaction.core.p
        world_p = Point3f(transform * Vec4f(local_p..., 1f0))

        # Transform normal to world space: n_world = normalize(transpose(inv_transform) * n_local)
        # For a 4x4 matrix, we use the upper-left 3x3 for direction transforms
        local_n = Vec3f(interaction.core.n)
        # transpose(inv_transform) is equivalent to inverse-transpose of transform
        inv_t_3x3 = Mat3f(inv_transform[1,1], inv_transform[2,1], inv_transform[3,1],
                          inv_transform[1,2], inv_transform[2,2], inv_transform[3,2],
                          inv_transform[1,3], inv_transform[2,3], inv_transform[3,3])
        world_n = Normal3f(normalize(inv_t_3x3 * local_n))

        # Transform shading normal similarly
        local_sn = Vec3f(interaction.shading.n)
        world_sn = Normal3f(normalize(inv_t_3x3 * local_sn))

        # Transform tangent/bitangent (these transform like directions, using the forward transform)
        t_3x3 = Mat3f(transform[1,1], transform[2,1], transform[3,1],
                      transform[1,2], transform[2,2], transform[3,2],
                      transform[1,3], transform[2,3], transform[3,3])
        # ∂p∂u and ∂p∂v are on SurfaceInteraction directly, not on core
        world_dpdu = normalize(t_3x3 * interaction.∂p∂u)
        world_dpdv = normalize(t_3x3 * interaction.∂p∂v)
        # Shading tangent/bitangent
        world_st = normalize(t_3x3 * interaction.shading.∂p∂u)
        world_sb = normalize(t_3x3 * interaction.shading.∂p∂v)

        # Reconstruct SurfaceInteraction with world-space values
        # Interaction fields: p, time, wo, n
        core = Interaction(world_p, interaction.core.time, interaction.core.wo, world_n)
        # ShadingInteraction fields: n, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v
        shading = ShadingInteraction(world_sn, world_st, world_sb, interaction.shading.∂n∂u, interaction.shading.∂n∂v)
        # SurfaceInteraction constructor: core, shading, uv, ∂p∂u, ∂p∂v, ∂n∂u, ∂n∂v, ∂u∂x, ∂u∂y, ∂v∂x, ∂v∂y, ∂p∂x, ∂p∂y
        interaction = SurfaceInteraction(
            core, shading, interaction.uv,
            world_dpdu, world_dpdv, interaction.∂n∂u, interaction.∂n∂v,
            interaction.∂u∂x, interaction.∂u∂y, interaction.∂v∂x, interaction.∂v∂y,
            interaction.∂p∂x, interaction.∂p∂y
        )
    end

    return true, triangle, interaction
end

# Intersect BVH - returns hit info, primitive, and SurfaceInteraction
@propagate_inbounds function intersect!(accel::BVH, ray::AbstractRay)
    hit_found, triangle, distance, bary_coords = closest_hit(accel, ray)

    if !hit_found
        return false, triangle, SurfaceInteraction()
    end

    # Convert to SurfaceInteraction
    interaction = triangle_to_surface_interaction(triangle, ray, bary_coords)

    # Return primitive so caller can access triangle.metadata (MaterialIndex)
    return true, triangle, interaction
end

# ============================================================================
# Scene Constructors
# ============================================================================

"""
    Scene(meshes, material_types, material_indices, materials::Tuple; lights=(), media=(), transforms=nothing)

Construct a Scene with multiple material types using TLAS.

# Arguments
- `meshes`: Vector of TriangleMesh geometries
- `material_types`: Vector{UInt8} specifying which tuple slot each mesh uses (1-based)
- `material_indices`: Vector{UInt32} specifying index within that material type's array
- `materials`: Tuple of material arrays, e.g., (Vector{MatteMaterial}, Vector{GlassMaterial})
- `lights`: Tuple or vector of lights (default empty)
- `media`: Tuple of media types (default empty)
- `transforms`: Optional vector of transforms per mesh
"""
function Scene(
    meshes::AbstractVector,
    material_types::AbstractVector{UInt8},
    material_indices::AbstractVector{<:Integer},
    materials::Tuple;
    lights::Union{Tuple,AbstractVector} = (),
    media::Tuple = (),
    transforms::Union{Nothing, AbstractVector{Mat4f}} = nothing
)
    scene, _handles = scene_with_handles(meshes, material_types, material_indices, materials; lights, media, transforms)
    return scene
end

"""
    scene_with_handles(meshes, material_types, material_indices, materials; lights=(), media=(), transforms=nothing)

Like `Scene(...)` but also returns instance handles for dynamic updates.
Returns `(scene::Scene, handles::Vector{InstanceHandle})`.

The handles can be used with `Raycore.update_transform!(tlas, handle, new_transform)`
followed by `Raycore.refit_tlas!(tlas)` for animation.
"""
function scene_with_handles(
    meshes::AbstractVector,
    material_types::AbstractVector{UInt8},
    material_indices::AbstractVector{<:Integer},
    materials::Tuple;
    lights::Union{Tuple,AbstractVector} = (),
    media::Tuple = (),
    transforms::Union{Nothing, AbstractVector{Mat4f}} = nothing
)
    # Create Instance objects with MaterialIndex metadata
    instances = if isnothing(transforms)
        [
            Instance(mesh; metadata=MaterialIndex(material_types[i], UInt32(material_indices[i])))
            for (i, mesh) in enumerate(meshes)
        ]
    else
        [
            Instance(mesh, transforms[i], MaterialIndex(material_types[i], UInt32(material_indices[i])))
            for (i, mesh) in enumerate(meshes)
        ]
    end
    tlas, handles = TLAS(instances)

    # Build MultiTypeVec for materials
    materials_mtv = MultiTypeVec(CPU())
    for mat_array in materials
        for mat in mat_array
            push!(materials_mtv, mat)
        end
    end

    # Build MultiTypeVec for media
    media_mtv = MultiTypeVec(CPU())
    for medium in media
        push!(media_mtv, medium)
    end

    scene = Scene(lights, tlas, materials_mtv, media_mtv)
    return scene, handles
end

"""
    Scene(mesh_material_pairs::Vector{<:Tuple}; lights=())

Construct a Scene from a vector of (mesh, material) pairs.
Automatically groups materials by type and assigns material_type/material_idx.

# Example
```julia
ground_mesh = TriangleMesh(ground_geom)
cloud_mesh = TriangleMesh(cloud_box_geom)
ground_mat = MatteMaterial(...)
cloud_vol = CloudVolume(data, extent)

scene = Scene([
    (ground_mesh, ground_mat),
    (cloud_mesh, cloud_vol),
]; lights=(PointLight(...),))
```

Also supports 3-tuples with transforms: `(mesh, material, transform::Mat4f)`.
"""
function Scene(mesh_material_pairs::Vector{<:Tuple}; lights::Union{Tuple,AbstractVector} = ())
    scene, _handles = scene_with_handles(mesh_material_pairs; lights)
    return scene
end

"""
    scene_with_handles(mesh_material_pairs::Vector{<:Tuple}; lights=())

Like `Scene(pairs)` but also returns instance handles for dynamic updates.
Returns `(scene::Scene, handles::Vector{InstanceHandle})`.

Automatically extracts media from `MediumInterface` materials and converts them
to `MediumInterfaceIdx` with proper indices for GPU dispatch.
"""
function scene_with_handles(mesh_material_pairs::Vector{<:Tuple}; lights::Union{Tuple,AbstractVector} = ())
    # Extract components - handle both 2-tuples and 3-tuples
    meshes = [pair[1] for pair in mesh_material_pairs]
    materials_list = [pair[2] for pair in mesh_material_pairs]
    has_transforms = length(mesh_material_pairs) > 0 && length(first(mesh_material_pairs)) >= 3
    transforms = has_transforms ? [Mat4f(pair[3]) for pair in mesh_material_pairs] : nothing

    # === Extract media from MediumInterface materials ===
    media_list = Any[]  # Unique media objects
    medium_to_index = Dict{Any, Int}()  # Medium object -> 1-based index

    for mat in materials_list
        if mat isa MediumInterface
            for medium in (mat.inside, mat.outside)
                if medium !== nothing && !haskey(medium_to_index, medium)
                    push!(media_list, medium)
                    medium_to_index[medium] = length(media_list)
                end
            end
        end
    end

    # Convert MediumInterface -> MediumInterfaceIdx with proper indices
    converted_materials = [to_indexed(mat, medium_to_index) for mat in materials_list]

    # Build media tuple (empty if no media)
    media_tuple = isempty(media_list) ? () : Tuple(media_list)

    # === Build materials tuple with converted materials ===
    # First pass: discover unique material types and their order
    type_to_slot = Dict{DataType, UInt8}()
    type_order = DataType[]

    for mat in converted_materials
        T = typeof(mat)
        if !haskey(type_to_slot, T)
            type_to_slot[T] = UInt8(length(type_to_slot) + 1)
            push!(type_order, T)
        end
    end

    # Second pass: count materials per type
    type_counts = Dict{DataType, Int}()
    for mat in converted_materials
        T = typeof(mat)
        type_counts[T] = get(type_counts, T, 0) + 1
    end

    # Create typed vectors and indices
    typed_vectors = Dict{DataType, Any}()
    type_current_idx = Dict{DataType, Int}()
    for T in type_order
        typed_vectors[T] = Vector{T}(undef, type_counts[T])
        type_current_idx[T] = 0
    end

    material_types = Vector{UInt8}(undef, length(converted_materials))
    material_indices = Vector{UInt32}(undef, length(converted_materials))

    for (i, mat) in enumerate(converted_materials)
        T = typeof(mat)
        slot = type_to_slot[T]
        type_current_idx[T] += 1
        idx = type_current_idx[T]
        typed_vectors[T][idx] = mat
        material_types[i] = slot
        material_indices[i] = UInt32(idx)
    end

    materials_arrays = [typed_vectors[type_order[i]] for i in 1:length(type_order)]
    materials_tuple = Tuple(materials_arrays)

    return scene_with_handles(meshes, material_types, material_indices, materials_tuple; lights, media=media_tuple, transforms)
end

include("filter.jl")
include("film.jl")


include("camera/camera.jl")
include("sampler/sampling.jl")
include("sampler/sampler.jl")
include("textures/mapping.jl")
include("textures/basic.jl")
include("textures/texture-ref.jl")
include("textures/environment_map.jl")
include("materials/uber-material.jl")
include("reflection/Reflection.jl")
include("materials/bsdf.jl")
include("materials/material.jl")
include("materials/volume.jl")
include("materials/emissive.jl")
include("materials/coated-diffuse.jl")
include("materials/mix-material.jl")
include("materials/thin-dielectric.jl")
include("materials/diffuse-transmission.jl")
include("materials/coated-conductor.jl")

# MediumIndex and MediumInterface (needed by spectral-eval.jl for BSDF forwarding)
include("materials/medium-interface.jl")

# Spectral rendering support (for PhysicalWavefront)
include("spectral/spectral.jl")
include("spectral/color.jl")
include("spectral/uplift.jl")
include("materials/spectral-eval.jl")

# Sobol sampler (needs mix_bits from spectral-eval.jl)
include("sampler/sobol_matrices.jl")
include("sampler/sobol.jl")

# Stratified sampler (needs murmur_hash_64a from spectral-eval.jl, sobol functions from sobol.jl)
include("sampler/stratified.jl")

include("primitive.jl")

# GeometricPrimitive convenience constructor for Scene
# Must be after primitive.jl which defines GeometricPrimitive
"""
    Scene(primitives::Vector{<:GeometricPrimitive}; lights=())

Construct a Scene from a vector of GeometricPrimitive objects.
Converts to (mesh, material) pairs and delegates to the tuple constructor.

# Example
```julia
s1 = GeometricPrimitive(mesh1, MatteMaterial(...))
s2 = GeometricPrimitive(mesh2, MirrorMaterial(...))
scene = Scene([s1, s2]; lights=(PointLight(...),))
```
"""
function Scene(primitives::Vector{<:GeometricPrimitive}; lights::Union{Tuple,AbstractVector} = ())
    pairs = [(p.shape, p.material) for p in primitives]
    return Scene(pairs; lights)
end

include("lights/emission.jl")
include("lights/light.jl")
include("lights/point.jl")
include("lights/spot.jl")
include("lights/directional.jl")
include("lights/sun.jl")
include("lights/sun_sky.jl")
include("lights/ambient.jl")
include("lights/environment.jl")
include("lights/light-sampler.jl")

include("integrators/sampler.jl")
include("integrators/sppm.jl")
include("integrators/fast-wavefront.jl")

# Unified work queue for wavefront integrators
include("integrators/workqueue.jl")

# PhysicalWavefront spectral path tracer
include("integrators/physical-wavefront/workitems.jl")
include("integrators/physical-wavefront/material-dispatch.jl")
include("integrators/physical-wavefront/lights.jl")
include("integrators/physical-wavefront/camera.jl")
include("integrators/physical-wavefront/intersection.jl")
include("integrators/physical-wavefront/material-eval.jl")
include("integrators/physical-wavefront/film-update.jl")

# VolPath volumetric path tracer
include("integrators/volpath/media.jl")
include("integrators/volpath/nanovdb.jl")
include("integrators/volpath/medium-dispatch.jl")
include("integrators/volpath/workitems.jl")
include("integrators/volpath/volpath-state.jl")
include("integrators/volpath/delta-tracking.jl")
include("integrators/volpath/medium-scatter.jl")
include("integrators/volpath/intersection.jl")
include("integrators/volpath/surface-eval.jl")
include("integrators/volpath/multi-material-eval.jl")
include("integrators/volpath/volpath.jl")

include("kernel-abstractions.jl")

# Postprocessing pipeline
include("postprocess.jl")

# Denoising
include("denoise.jl")

# include("model_loader.jl")

end
