module Hikari

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

abstract type Spectrum end
abstract type Light end
abstract type Material end
abstract type BxDF end
abstract type Integrator end

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

const ENABLE_INBOUNDS = true

macro _inbounds(ex)
    if ENABLE_INBOUNDS
        esc(:(@inbounds $ex))
    else
        esc(ex)
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

@inline maybe_copy(v::Maybe)::Maybe = v isa Nothing ? v : copy(v)

@inline function concentric_sample_disk(u::Point2f)::Point2f
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

@inline uniform_sphere_pdf()::Float32 = 1f0 / (4f0 * π)

@inline function uniform_cone_pdf(cosθ_max::Float32)::Float32
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
@inline cos_θ(w::Vec3f) = w[3]
@inline sin_θ2(w::Vec3f) = max(0f0, 1f0 - cos_θ(w) * cos_θ(w))
@inline sin_θ(w::Vec3f) = √(sin_θ2(w))
@inline tan_θ(w::Vec3f) = sin_θ(w) / cos_θ(w)

@inline function cos_ϕ(w::Vec3f)
    sinθ = sin_θ(w)
    sinθ ≈ 0f0 ? 1f0 : clamp(w[1] / sinθ, -1f0, 1f0)
end
@inline function sin_ϕ(w::Vec3f)
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
@inline face_forward(n, v) = (n ⋅ v) < 0 ? -n : n

include("spectrum.jl")
include("surface_interaction.jl")

# Abstract scene type for dispatch - allows both mutable and immutable implementations
abstract type AbstractScene end

# Mutable scene for CPU use - allows updating lights after creation
mutable struct Scene{P, L<:NTuple{N, Light} where N} <: AbstractScene
    lights::L
    aggregate::P
    bound::Bounds3
    world_center::Point3f
    world_radius::Float32
end

# Immutable scene for e.g. GPU use - required for GPU kernels (must be bitstype)
struct ImmutableScene{P, L<:NTuple{N, Light} where N} <: AbstractScene
    lights::L
    aggregate::P
    bound::Bounds3
    world_center::Point3f
    world_radius::Float32
end

function Scene(
        lights::Union{Tuple,AbstractVector}, aggregate::P,
    ) where {P}
    ltuple = Tuple(lights)
    bounds = world_bound(aggregate)
    world_center, world_radius = bounding_sphere(bounds)
    Scene{P,typeof(ltuple)}(ltuple, aggregate, bounds, world_center, world_radius)
end

# Convert Scene to ImmutableScene for GPU rendering
ImmutableScene(s::Scene) = ImmutableScene(s.lights, s.aggregate, s.bound, s.world_center, s.world_radius)
ImmutableScene(s::ImmutableScene) = s  # Already immutable

# Common interface for both scene types
@inline function intersect!(scene::AbstractScene, ray::AbstractRay)
    intersect!(scene.aggregate, ray)
end

@inline function intersect_p(scene::AbstractScene, ray::AbstractRay)
    intersect_p(scene.aggregate, ray)
end

# Pretty printing for Scene
function Base.show(io::IO, ::MIME"text/plain", scene::Scene)
    n_lights = length(scene.lights)

    println(io, "Scene:")
    println(io, "  Lights:     ", n_lights)
    if n_lights > 0
        for (i, light) in enumerate(scene.lights)
            println(io, "    [", i, "] ", typeof(light).name.name)
        end
    end
    println(io, "  Aggregate:  ", typeof(scene.aggregate).name.name)
    print(io,   "  Bounds:     ", scene.bound.p_min, " to ", scene.bound.p_max)
end

function Base.show(io::IO, scene::Scene)
    if get(io, :compact, false)
        n_lights = length(scene.lights)
        print(io, "Scene(lights=", n_lights, ", aggregate=", typeof(scene.aggregate).name.name, ")")
    else
        show(io, MIME("text/plain"), scene)
    end
end

# spawn_ray functions are now in Raycore, but we need versions for our SurfaceInteraction
# Raycore only has spawn_ray for its simpler Interaction type

# We'll create a MaterialScene wrapper below

"""
    MaterialIndex

Metadata type for triangles that stores material lookup information.
- `material_type`: Which tuple slot (1-based) in the materials tuple
- `material_idx`: Index within that material type's array
"""
struct MaterialIndex
    material_type::UInt8
    material_idx::UInt32
end

# MaterialScene: wraps any accelerator (BVH, TLAS, etc.) with materials stored as a tuple of arrays
# Each material type gets its own array, indexed by triangle.metadata::MaterialIndex
struct MaterialScene{Accel, Materials<:Tuple}
    accel::Accel  # BVH, TLAS, or any accelerator supporting closest_hit/any_hit
    materials::Materials  # Tuple of material arrays, e.g., (Vector{MatteMaterial}, Vector{GlassMaterial})
end

@inline world_bound(ms::MaterialScene) = world_bound(ms.accel)

# Generated function for type-stable material dispatch
# Returns the material from the appropriate tuple slot
@generated function get_material(materials::NTuple{N,Any}, idx::MaterialIndex) where N
    branches = [quote
        if idx.material_type === UInt8($i)
            return @inbounds materials[$i][idx.material_idx]
        end
    end for i in 1:N]
    # Return first material type as fallback (GPU-compatible, no error() call)
    # This should never happen in practice if material indices are valid
    quote
        $(branches...)
        return @inbounds materials[1][1]
    end
end

# Type-stable shade dispatch - each material type implements shade(material, ray, si, scene, beta, depth, max_depth)
# IMPORTANT: Type annotations on ray, si, scene, beta prevent argument boxing in generated code
@inline @generated function shade_material(
    materials::NTuple{N}, idx::MaterialIndex,
    ray::RayDifferentials, si::SurfaceInteraction, scene::S, beta::RGBSpectrum, depth::Int32, max_depth::Int32
) where {N, S<:AbstractScene}
    branches = [quote
        @inbounds if idx.material_type === UInt8($i)
            return @inline shade(materials[$i][idx.material_idx], ray, si, scene, beta, depth, max_depth)
        end
    end for i in 1:N]
    quote
        $(branches...)
        return RGBSpectrum(0f0)
    end
end

# Type-stable bounce ray generation - materials implement sample_bounce(material, ray, si, scene, beta, depth)
# IMPORTANT: Type annotations prevent argument boxing in generated code
@inline @generated function sample_material_bounce(
    materials::NTuple{N}, idx::MaterialIndex,
    ray::RayDifferentials, si::SurfaceInteraction, scene::S, beta::RGBSpectrum, depth::Int32
) where {N, S<:AbstractScene}
    branches = [quote
        @inbounds if idx.material_type === UInt8($i)
            return @inline sample_bounce(materials[$i][idx.material_idx], ray, si, scene, beta, depth)
        end
    end for i in 1:N]
    quote
        $(branches...)
        # No bounce
        return (false, ray, RGBSpectrum(0f0), Int32(0))
    end
end

# Calculate partial derivatives for texture mapping
function partial_derivatives(vs::AbstractVector{Point3f}, uv::AbstractVector{Point2f})
    δuv_13, δuv_23 = uv[1] - uv[3], uv[2] - uv[3]
    δp_13, δp_23 = Vec3f(vs[1] - vs[3]), Vec3f(vs[2] - vs[3])
    det = δuv_13[1] * δuv_23[2] - δuv_13[2] * δuv_23[1]
    if det ≈ 0f0
        v = normalize((vs[3] - vs[1]) × (vs[2] - vs[1]))
        _, ∂p∂u, ∂p∂v = coordinate_system(Vec3f(v))
        return ∂p∂u, ∂p∂v
    end
    inv_det = 1f0 / det
    ∂p∂u = Vec3f(δuv_23[2] * δp_13 - δuv_13[2] * δp_23) * inv_det
    ∂p∂v = Vec3f(-δuv_23[1] * δp_13 + δuv_13[1] * δp_23) * inv_det
    return ∂p∂u, ∂p∂v
end

# Convert Raycore.Triangle intersection result to Trace SurfaceInteraction
@inline function triangle_to_surface_interaction(triangle::Triangle, ray::AbstractRay, bary_coords::StaticVector{3,Float32})::SurfaceInteraction
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


# Intersect function for MaterialScene - returns hit info, primitive, and SurfaceInteraction
# The primitive (triangle) contains material_type and material_idx for dispatch
@inline function intersect!(ms::MaterialScene, ray::AbstractRay)
    accel = ms.accel

    # Handle TLAS (instanced) vs BVH (non-instanced) differently
    if accel isa TLAS
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
    else
        # Non-instanced BVH path (original behavior)
        hit_found, triangle, distance, bary_coords = closest_hit(accel, ray)

        if !hit_found
            return false, triangle, SurfaceInteraction()
        end

        # Convert to SurfaceInteraction
        interaction = triangle_to_surface_interaction(triangle, ray, bary_coords)

        # Return primitive so caller can access triangle.metadata (MaterialIndex)
        return true, triangle, interaction
    end
end

@inline function intersect_p(ms::MaterialScene, ray::AbstractRay)
    hit_found, _, _, _ = any_hit(ms.accel, ray)
    return hit_found
end

# Pretty printing for MaterialScene
function Base.show(io::IO, ::MIME"text/plain", ms::MaterialScene)
    accel = ms.accel
    accel_type = nameof(typeof(accel))
    n_material_types = length(ms.materials)
    n_materials_total = sum(length, ms.materials)
    bounds = world_bound(accel)

    println(io, "MaterialScene ($accel_type):")

    # Show accelerator-specific stats
    if accel isa TLAS
        n_instances = length(accel.instances)
        n_geometries = length(accel.blas_array)
        n_triangles = sum(blas -> length(blas.primitives), accel.blas_array)
        n_top_nodes = length(accel.nodes)
        n_top_leaves = count(node -> Raycore.is_leaf(node), accel.nodes)
        n_top_interior = n_top_nodes - n_top_leaves

        println(io, "  Geometries:     ", n_geometries)
        println(io, "  Instances:      ", n_instances)
        println(io, "  Triangles:      ", n_triangles, " (in BLAS)")
        println(io, "  TLAS nodes:     ", n_top_nodes, " (", n_top_interior, " interior, ", n_top_leaves, " leaves)")
    elseif accel isa BVH
        n_triangles = length(accel.primitives)
        n_nodes = length(accel.nodes)
        println(io, "  Triangles:      ", n_triangles)
        println(io, "  BVH nodes:      ", n_nodes)
    else
        # Generic fallback
        println(io, "  Accelerator:    ", accel_type)
    end

    println(io, "  Material types: ", n_material_types)
    println(io, "  Total materials:", n_materials_total)
    print(io,   "  Bounds:         ", bounds.p_min, " to ", bounds.p_max)
end

function Base.show(io::IO, ms::MaterialScene)
    if get(io, :compact, false)
        accel = ms.accel
        accel_type = nameof(typeof(accel))
        n_material_types = length(ms.materials)

        if accel isa TLAS
            n_instances = length(accel.instances)
            n_geometries = length(accel.blas_array)
            print(io, "MaterialScene($accel_type, geoms=$n_geometries, instances=$n_instances, materials=$n_material_types)")
        elseif accel isa BVH
            n_triangles = length(accel.primitives)
            print(io, "MaterialScene($accel_type, triangles=$n_triangles, materials=$n_material_types)")
        else
            print(io, "MaterialScene($accel_type, materials=$n_material_types)")
        end
    else
        show(io, MIME("text/plain"), ms)
    end
end

"""
    MaterialScene(meshes, material_types, material_indices, materials::Tuple)

Construct a MaterialScene with multiple material types using TLAS.

# Arguments
- `meshes`: Vector of TriangleMesh geometries
- `material_types`: Vector{UInt8} specifying which tuple slot each mesh uses (1-based)
- `material_indices`: Vector{UInt32} specifying index within that material type's array
- `materials`: Tuple of material arrays, e.g., (Vector{MatteMaterial}, Vector{GlassMaterial})

# Example
```julia
# Create materials
uber_materials = [MatteMaterial(...), MirrorMaterial(...)]
volumes = [CloudVolume(...)]

# Create meshes
ground_mesh = TriangleMesh(ground_geometry)
cloud_box_mesh = TriangleMesh(box_geometry)

# Build scene: ground uses MatteMaterial[1], cloud uses CloudVolume[1]
scene = MaterialScene(
    [ground_mesh, cloud_box_mesh],
    UInt8[1, 2],           # material types (1=uber, 2=volume)
    UInt32[1, 1],          # indices within each type
    (uber_materials, volumes)
)
```
"""
function MaterialScene(
    meshes::AbstractVector,
    material_types::AbstractVector{UInt8},
    material_indices::AbstractVector{<:Integer},
    materials::Tuple;
    transforms::Union{Nothing, AbstractVector{Mat4f}} = nothing
)
    scene, _handles = material_scene_with_handles(meshes, material_types, material_indices, materials; transforms)
    return scene
end

"""
    material_scene_with_handles(meshes, material_types, material_indices, materials; transforms=nothing)

Like `MaterialScene(...)` but also returns instance handles for dynamic updates.
Returns `(scene::MaterialScene, handles::Vector{InstanceHandle})`.

The handles can be used with `Raycore.update_transform!(tlas, handle, new_transform)`
followed by `Raycore.refit_tlas!(tlas)` for animation.
"""
function material_scene_with_handles(
    meshes::AbstractVector,
    material_types::AbstractVector{UInt8},
    material_indices::AbstractVector{<:Integer},
    materials::Tuple;
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
    return MaterialScene(tlas, materials), handles
end

"""
    MaterialScene(mesh_material_pairs::Vector{<:Tuple{Any, <:Material}})

Construct a MaterialScene from a vector of (mesh, material) pairs.
Automatically groups materials by type and assigns material_type/material_idx.

# Example
```julia
ground_mesh = TriangleMesh(ground_geom)
cloud_mesh = TriangleMesh(cloud_box_geom)
ground_mat = MatteMaterial(...)
cloud_vol = CloudVolume(data, extent)

scene = MaterialScene([
    (ground_mesh, ground_mat),
    (cloud_mesh, cloud_vol),
])
```

Also supports 3-tuples with transforms: `(mesh, material, transform::Mat4f)`.
When transforms are provided, the returned scene's TLAS uses instanced transforms
which can be updated via `Raycore.update_transform!` and `Raycore.refit_tlas!`.
"""
function MaterialScene(mesh_material_pairs::Vector{<:Tuple})
    # Extract components - handle both 2-tuples and 3-tuples
    meshes = [pair[1] for pair in mesh_material_pairs]
    materials_list = [pair[2] for pair in mesh_material_pairs]
    has_transforms = length(mesh_material_pairs) > 0 && length(first(mesh_material_pairs)) >= 3
    transforms = has_transforms ? [Mat4f(pair[3]) for pair in mesh_material_pairs] : nothing

    # First pass: discover unique material types and their order
    type_to_slot = Dict{DataType, UInt8}()
    type_order = DataType[]  # Keep track of order types were discovered

    for mat in materials_list
        T = typeof(mat)
        if !haskey(type_to_slot, T)
            type_to_slot[T] = UInt8(length(type_to_slot) + 1)
            push!(type_order, T)
        end
    end

    # Second pass: count materials per type to pre-allocate
    type_counts = Dict{DataType, Int}()
    for mat in materials_list
        T = typeof(mat)
        type_counts[T] = get(type_counts, T, 0) + 1
    end

    # Create properly typed vectors for each material type
    # We use a function barrier to ensure type stability
    function create_typed_vectors(type_order, type_counts, materials_list, type_to_slot)
        # Create typed vectors
        typed_vectors = Dict{DataType, Any}()
        type_current_idx = Dict{DataType, Int}()
        for T in type_order
            typed_vectors[T] = Vector{T}(undef, type_counts[T])
            type_current_idx[T] = 0
        end

        material_types = Vector{UInt8}(undef, length(materials_list))
        material_indices = Vector{UInt32}(undef, length(materials_list))

        # Fill in materials
        for (i, mat) in enumerate(materials_list)
            T = typeof(mat)
            slot = type_to_slot[T]
            type_current_idx[T] += 1
            idx = type_current_idx[T]
            typed_vectors[T][idx] = mat
            material_types[i] = slot
            material_indices[i] = UInt32(idx)
        end

        # Build tuple in slot order
        materials_arrays = [typed_vectors[type_order[i]] for i in 1:length(type_order)]
        return Tuple(materials_arrays), material_types, material_indices
    end

    materials_tuple, material_types, material_indices = create_typed_vectors(
        type_order, type_counts, materials_list, type_to_slot
    )

    # Use the first constructor which builds the TLAS
    return MaterialScene(meshes, material_types, material_indices, materials_tuple; transforms)
end

"""
    material_scene_with_handles(mesh_material_pairs::Vector{<:Tuple})

Like `MaterialScene(pairs)` but also returns instance handles for dynamic updates.
Accepts 2-tuples `(mesh, material)` or 3-tuples `(mesh, material, transform)`.
Returns `(scene::MaterialScene, handles::Vector{InstanceHandle})`.
"""
function material_scene_with_handles(mesh_material_pairs::Vector{<:Tuple})
    # Extract components - handle both 2-tuples and 3-tuples
    meshes = [pair[1] for pair in mesh_material_pairs]
    materials_list = [pair[2] for pair in mesh_material_pairs]
    has_transforms = length(mesh_material_pairs) > 0 && length(first(mesh_material_pairs)) >= 3
    transforms = has_transforms ? [Mat4f(pair[3]) for pair in mesh_material_pairs] : nothing

    # First pass: discover unique material types and their order
    type_to_slot = Dict{DataType, UInt8}()
    type_order = DataType[]

    for mat in materials_list
        T = typeof(mat)
        if !haskey(type_to_slot, T)
            type_to_slot[T] = UInt8(length(type_to_slot) + 1)
            push!(type_order, T)
        end
    end

    # Second pass: count materials per type
    type_counts = Dict{DataType, Int}()
    for mat in materials_list
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

    material_types = Vector{UInt8}(undef, length(materials_list))
    material_indices = Vector{UInt32}(undef, length(materials_list))

    for (i, mat) in enumerate(materials_list)
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

    return material_scene_with_handles(meshes, material_types, material_indices, materials_tuple; transforms)
end

include("filter.jl")
include("film.jl")


include("camera/camera.jl")
include("sampler/sampling.jl")
include("sampler/sampler.jl")
include("textures/mapping.jl")
include("textures/basic.jl")
include("textures/environment_map.jl")
include("materials/uber-material.jl")
include("reflection/Reflection.jl")
include("materials/bsdf.jl")
include("materials/material.jl")
include("materials/volume.jl")
include("materials/emissive.jl")

# Spectral rendering support (for PhysicalWavefront)
include("spectral/spectral.jl")
include("spectral/color.jl")
include("spectral/uplift.jl")
include("materials/spectral-eval.jl")
include("primitive.jl")

# GeometricPrimitive convenience constructor for MaterialScene
# Must be after primitive.jl which defines GeometricPrimitive
"""
    MaterialScene(primitives::Vector{<:GeometricPrimitive})

Construct a MaterialScene from a vector of GeometricPrimitive objects.
Converts to (mesh, material) pairs and delegates to the tuple constructor.

# Example
```julia
s1 = GeometricPrimitive(mesh1, MatteMaterial(...))
s2 = GeometricPrimitive(mesh2, MirrorMaterial(...))
scene = MaterialScene([s1, s2])
```
"""
function MaterialScene(primitives::Vector{<:GeometricPrimitive})
    pairs = [(p.shape, p.material) for p in primitives]
    return MaterialScene(pairs)
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

include("integrators/sampler.jl")
include("integrators/sppm.jl")
include("integrators/fast-wavefront.jl")

# PhysicalWavefront spectral path tracer
include("integrators/physical-wavefront/workitems.jl")
include("integrators/physical-wavefront/workqueue.jl")
include("integrators/physical-wavefront/material-dispatch.jl")
include("integrators/physical-wavefront/lights.jl")
include("integrators/physical-wavefront/camera.jl")
include("integrators/physical-wavefront/intersection.jl")
include("integrators/physical-wavefront/material-eval.jl")
include("integrators/physical-wavefront/film-update.jl")
include("integrators/physical-wavefront/physical-wavefront.jl")

include("kernel-abstractions.jl")

# Postprocessing pipeline
include("postprocess.jl")

# Denoising
include("denoise.jl")

# include("model_loader.jl")

end
