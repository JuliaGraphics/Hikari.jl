Base.@kwdef mutable struct Ray <: AbstractRay
    o::Point3f
    d::Vec3f
    t_max::Float32 = Inf32
    time::Float32 = 0f0
end

Base.@kwdef mutable struct RayDifferentials <: AbstractRay
    o::Point3f
    d::Vec3f
    t_max::Float32 = Inf32
    time::Float32 = 0f0

    has_differentials::Bool = false
    rx_origin::Point3f = zeros(Point3f)
    ry_origin::Point3f = zeros(Point3f)
    rx_direction::Vec3f = zeros(Vec3f)
    ry_direction::Vec3f = zeros(Vec3f)
end

@inline function RayDifferentials(r::Ray)::RayDifferentials
    RayDifferentials(o = r.o, d = r.d, t_max = r.t_max, time = r.time)
end

@inline function set_direction!(r::AbstractRay, d::Vec3f)
    r.d = Vec3f([i ≈ 0f0 ? 0f0 : i for i in d])
end

@inline check_direction!(r::AbstractRay) = set_direction!(r, r.d)

function (r::Union{Ray,RayDifferentials})(t::Number)
    r.o + r.d * t
end

function scale_differentials!(rd::RayDifferentials, s::Float32)
    rd.rx_origin = rd.o + (rd.rx_origin - rd.o) * s
    rd.ry_origin = rd.o + (rd.ry_origin - rd.o) * s
    rd.rx_direction = rd.d + (rd.rx_direction - rd.d) * s
    rd.rx_direction = rd.d + (rd.ry_direction - rd.d) * s
end
