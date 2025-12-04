module Trace

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
import Raycore: AccelPrimitive, BVH, world_bound, closest_hit, any_hit
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

"""
Reflect `wo` about `n`.
"""
@inline reflect(wo::Vec3f, n::Vec3f) = -wo + 2f0 * (wo ⋅ n) * n

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

function coordinate_system(v1::Vec3f)
    if abs(v1[1]) > abs(v1[2])
        v2 = Vec3f(-v1[3], 0, v1[1]) / sqrt(v1[1] * v1[1] + v1[3] * v1[3])
    else
        v2 = Vec3f(0, v1[3], -v1[2]) / sqrt(v1[2] * v1[2] + v1[3] * v1[3])
    end
    v1, v2, v1 × v2
end

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

struct Scene{P, L<:NTuple{N, Light} where N}
    lights::L
    aggregate::P
    bound::Bounds3
end

function Scene(
        lights::Union{Tuple,AbstractVector}, aggregate::P,
    ) where {P}
    # TODO preprocess for lights
    ltuple = Tuple(lights)
    Scene{P,typeof(ltuple)}(ltuple, aggregate, world_bound(aggregate))
end

@inline function intersect!(scene::Scene, ray::AbstractRay)
    intersect!(scene.aggregate, ray)
end

@inline function intersect_p(scene::Scene, ray::AbstractRay)
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

# MaterialScene: wraps Raycore.BVH with materials
# Raycore.BVH only stores triangles, we map triangle.material_idx -> material
struct MaterialScene{BVH<:AccelPrimitive, M<:AbstractVector{<:Material}}
    bvh::BVH
    materials::M
end

@inline world_bound(ms::MaterialScene) = world_bound(ms.bvh)

# Convert Raycore.Triangle intersection result to Trace SurfaceInteraction
function triangle_to_surface_interaction(triangle::Triangle, ray::AbstractRay, bary_coords::StaticVector{3,Float32})::SurfaceInteraction
    # Get triangle data
    verts = Raycore.vertices(triangle)
    tex_coords = Raycore.uvs(triangle)

    # Calculate partial derivatives
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
        ∂p∂u, ∂p∂v
    end

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


# Intersect function for MaterialScene - returns material and SurfaceInteraction
@inline function intersect!(ms::MaterialScene, ray::AbstractRay)
    hit_found, triangle, distance, bary_coords = closest_hit(ms.bvh, ray)

    if !hit_found
        return false, NoMaterial(), SurfaceInteraction()
    end

    # Convert to SurfaceInteraction
    interaction = triangle_to_surface_interaction(triangle, ray, bary_coords)

    # Get material from triangle's material_idx
    material = ms.materials[triangle.material_idx]

    return true, material, interaction
end

@inline function intersect_p(ms::MaterialScene, ray::AbstractRay)
    hit_found, _, _, _ = any_hit(ms.bvh, ray)
    return hit_found
end

# Pretty printing for MaterialScene
function Base.show(io::IO, ::MIME"text/plain", ms::MaterialScene)
    n_triangles = length(ms.bvh.primitives)
    n_materials = length(ms.materials)
    n_nodes = length(ms.bvh.nodes)
    bounds = world_bound(ms.bvh)

    # Count leaf vs interior nodes
    n_leaves = count(node -> !node.is_interior, ms.bvh.nodes)
    n_interior = n_nodes - n_leaves

    println(io, "MaterialScene:")
    println(io, "  Triangles:  ", n_triangles)
    println(io, "  Materials:  ", n_materials)
    println(io, "  BVH nodes:  ", n_nodes, " (", n_interior, " interior, ", n_leaves, " leaves)")
    println(io, "  Bounds:     ", bounds.p_min, " to ", bounds.p_max)
    print(io,   "  Max prims:  ", Int(ms.bvh.max_node_primitives), " per leaf")
end

function Base.show(io::IO, ms::MaterialScene)
    if get(io, :compact, false)
        n_triangles = length(ms.bvh.primitives)
        n_materials = length(ms.materials)
        n_nodes = length(ms.bvh.nodes)
        n_leaves = count(node -> !node.is_interior, ms.bvh.nodes)
        n_interior = n_nodes - n_leaves
        print(io, "MaterialScene(")
        print(io, "triangles=", n_triangles, ", ")
        print(io, "materials=", n_materials, ", ")
        print(io, "nodes=", n_nodes, " (", n_interior, " interior, ", n_leaves, " leaves)")
        print(io, ")")
    else
        show(io, MIME("text/plain"), ms)
    end
end

# Helper function for basic-scene.jl compatibility
function no_material_bvh(geometric_primitives::Vector)
    meshes = [gp.shape for gp in geometric_primitives]
    materials = [gp.material for gp in geometric_primitives]

    # Build BVH with material indices
    bvh = BVH(meshes)

    return MaterialScene(bvh, materials)
end

include("filter.jl")
include("film.jl")


include("camera/camera.jl")
include("sampler/sampling.jl")
include("sampler/sampler.jl")
include("textures/mapping.jl")
include("textures/basic.jl")
include("materials/uber-material.jl")
include("reflection/Reflection.jl")
include("materials/bsdf.jl")
include("materials/material.jl")
include("primitive.jl")

include("lights/emission.jl")
include("lights/light.jl")
include("lights/point.jl")
include("lights/spot.jl")
include("lights/directional.jl")
include("lights/ambient.jl")
include("lights/environment.jl")

include("integrators/sampler.jl")
include("integrators/sppm.jl")
include("kernel-abstractions.jl")

# include("model_loader.jl")

end
