module Trace

# using Assimp
import FileIO
using ImageCore
using ImageIO
using GeometryBasics
using LinearAlgebra
using StaticArrays
using ProgressMeter

abstract type AbstractRay end
abstract type Spectrum end
abstract type AbstractShape end
abstract type Primitive end
abstract type Light end
abstract type Material end
abstract type BxDF end
abstract type Integrator end

const Radiance = Val{:Radiance}
const Importance = Val{:Importance}
const TransportMode = Union{Radiance,Importance}

const DO_ASSERTS = false
macro real_assert(expr, msg="")
    if DO_ASSERTS
        esc(:(@assert $expr $msg))
    else
        return :()
    end
end

# GeometryBasics.@fixed_vector Normal StaticVector
include("typeNormal3f.jl")

const Maybe{T} = Union{T,Nothing}

function get_progress_bar(n::Integer, desc::String = "Progress")
    Progress(
        n, desc = desc, dt = 1,
        barglyphs = BarGlyphs("[=> ]"), barlen = 50, color = :white,
    )
end

@inline maybe_copy(v::Maybe)::Maybe = v isa Nothing ? v : copy(v)

function concentric_sample_disk(u::Point2f)::Point2f
    # Map uniform random numbers to [-1, 1].
    offset = 2f0 * u - Vec2f(1f0)
    # Handle degeneracy at the origin.
    offset[1] ≈ 0 && offset[2] ≈ 0 && return Point2f(0)
    if abs(offset[1]) > abs(offset[2])
        r = offset[1]
        θ = (offset[2] / offset[1]) * π / 4f0
    else
        r = offset[2]
        θ = π / 2f0 - (offset[1] / offset[2]) * π / 4f0
    end
    r * Point2f(cos(θ), sin(θ))
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

include("ray.jl")
include("bounds.jl")
include("transformations.jl")
include("spectrum.jl")
include("surface_interaction.jl")

struct Scene
    lights::Vector{L} where L<:Light
    aggregate::P where P<:Primitive
    bound::Bounds3

    function Scene(
        lights::Vector{L}, aggregate::P,
    ) where L<:Light where P<:Primitive
        # TODO preprocess for lights
        new(lights, aggregate, world_bound(aggregate))
    end
end

@inline function intersect!(scene::Scene, ray::Union{Ray,RayDifferentials})
    intersect!(scene.aggregate, ray)
end
@inline function intersect_p(scene::Scene, ray::Union{Ray,RayDifferentials})
    intersect_p(scene.aggregate, ray)
end

@inline function spawn_ray(
    p0::Interaction, p1::Interaction, δ::Float32 = 1f-6,
)::Ray
    direction = p1.p - p0.p
    origin = p0.p .+ δ .* direction
    Ray(o = origin, d = direction, time = p0.time)
end
@inline function spawn_ray(p0::SurfaceInteraction, p1::Interaction)::Ray
    spawn_ray(p0.core, p1)
end
@inline function spawn_ray(
    si::SurfaceInteraction, direction::Vec3f, δ::Float32 = 1f-6,
)::Ray
    origin = si.core.p .+ δ .* direction
    Ray(o = origin, d = direction, time = si.core.time)
end

include("shapes/Shape.jl")
include("primitive.jl")
include("accel/bvh.jl")

include("filter.jl")
include("film.jl")
include("reflection/Reflection.jl")

include("camera/camera.jl")
include("sampler/sampling.jl")
include("sampler/sampler.jl")
include("textures/mapping.jl")
include("textures/basic.jl")
include("materials/bsdf.jl")
include("materials/material.jl")

include("lights/emission.jl")
include("lights/light.jl")
include("lights/point.jl")
include("lights/spot.jl")
include("lights/directional.jl")

include("integrators/sampler.jl")
include("integrators/sppm.jl")

# include("model_loader.jl")

end
