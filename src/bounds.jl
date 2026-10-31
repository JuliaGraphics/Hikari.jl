struct Bounds2
    p_min::Point2f
    p_max::Point2f
end

struct Bounds3
    p_min::Point3f
    p_max::Point3f
end

# By default -- create bounds in invalid configuraiton.
Bounds2() = Bounds2(Point2f(Inf32), Point2f(-Inf32))
Bounds3() = Bounds3(Point3f(Inf32), Point3f(-Inf32))
Bounds2(p::Point2f) = Bounds2(p, p)
Bounds3(p::Point3f) = Bounds3(p, p)
Bounds2c(p1::Point2f, p2::Point2f) = Bounds2(min.(p1, p2), max.(p1, p2))
Bounds3c(p1::Point3f, p2::Point3f) = Bounds3(min.(p1, p2), max.(p1, p2))

function Base.:(==)(b1::Union{Bounds2,Bounds3}, b2::Union{Bounds2,Bounds3})
    b1.p_min == b2.p_min && b1.p_max == b2.p_max
end
function Base.:≈(b1::Union{Bounds2,Bounds3}, b2::Union{Bounds2,Bounds3})
    b1.p_min ≈ b2.p_min && b1.p_max ≈ b2.p_max
end
function Base.getindex(b::Union{Bounds2,Bounds3}, i::Integer)
    i == 1 && return b.p_min
    i == 2 && return b.p_max
    error("Invalid index `$i`. Only `1` & `2` are valid.")
end
function is_valid(b::Bounds3)::Bool
    all(b.p_min .!= Inf32) && all(b.p_max .!= -Inf32)
end

function Base.length(b::Bounds2)::Int64
    δ = ceil.(b.p_max .- b.p_min .+ 1f0)
    Int64(δ[1] * δ[2])
end

function Base.iterate(
    b::Bounds2, i::Integer = 1,
)::Union{Nothing,Tuple{Point2f,Integer}}
    i > length(b) && return nothing

    j = i - 1
    δ = b.p_max .- b.p_min .+ 1f0
    b.p_min .+ Point2f(j % δ[1], j ÷ δ[1]), i + 1
end

# Index through 8 corners.
function corner(b::Bounds3, c::Integer)
    c -= 1
    Point3f(
        b[(c&1)+1][1],
        b[(c & 2) != 0 ? 2 : 1][2],
        b[(c & 4) != 0 ? 2 : 1][3],
    )
end

function Base.union(b1::B, b2::B) where B<:Union{Bounds2,Bounds3}
    B(min.(b1.p_min, b2.p_min), max.(b1.p_max, b2.p_max))
end

function Base.intersect(b1::B, b2::B) where B<:Union{Bounds2,Bounds3}
    B(max.(b1.p_min, b2.p_min), min.(b1.p_max, b2.p_max))
end

function overlaps(b1::Bounds3, b2::Bounds3)
    all(b1.p_max .>= b2.p_min) && all(b1.p_min .<= b2.p_max)
end

function inside(b::Bounds3, p::Point3f)
    all(p .>= b.p_min) && all(p .<= b.p_max)
end

function inside_exclusive(b::Bounds3, p::Point3f)
    all(p .>= b.p_min) && all(p .< b.p_max)
end

expand(b::Bounds3, δ::Float32) = Bounds3(b.p_min .- δ, b.p_max .+ δ)
diagonal(b::Union{Bounds2,Bounds3}) = b.p_max - b.p_min

function surface_area(b::Bounds3)
    d = diagonal(b)
    2 * (d[1] * d[2] + d[1] * d[3] + d[2] * d[3])
end

function area(b::Bounds2)
    δ = b.p_max .- b.p_min
    δ[1] * δ[2]
end

@inline function sides(b::Union{Bounds2,Bounds3})
    [abs(b1 - b0) for (b1, b0) in zip(b.p_max, b.p_min)]
end

@inline function inclusive_sides(b::Union{Bounds2,Bounds3})
    [abs(b1 - (b0 - 1f0)) for (b1, b0) in zip(b.p_max, b.p_min)]
end

function volume(b::Bounds3)
    d = diagonal(b)
    d[1] * d[2] * d[3]
end

"""
Return index of the longest axis.
Useful for deciding which axis to subdivide,
when building ray-tracing acceleration structures.

1 - x, 2 - y, 3 - z.
"""
function maximum_extent(b::Bounds3)
    d = diagonal(b)
    if d[1] > d[2] && d[1] > d[3]
        return 1
    elseif d[2] > d[3]
        return 2
    end
    return 3
end

lerp(v1::Float32, v2::Float32, t::Float32) = (1 - t) * v1 + t * v2
lerp(p0::Point3f, p1::Point3f, t::Float32) = (1 - t) .* p0 .+ t .* p1
# Linearly interpolate point between the corners of the bounds.
lerp(b::Bounds3, p::Point3f) = lerp.(p, b.p_min, b.p_max)

distance(p1::Point3f, p2::Point3f) = norm(p1 - p2)
function distance_squared(p1::Point3f, p2::Point3f)
    p = p1 - p2
    p ⋅ p
end

"""Get offset of a point from the minimum point of the bounds."""
function offset(b::Bounds3, p::Point3f)
    o = p - b.p_min
    g = b.p_max .> b.p_min
    !any(g) && return o
    Point3f(
        o[1] / (g[1] ? b.p_max[1] - b.p_min[1] : 1f0),
        o[2] / (g[2] ? b.p_max[2] - b.p_min[2] : 1f0),
        o[3] / (g[3] ? b.p_max[3] - b.p_min[3] : 1f0),
    )
end

function bounding_sphere(b::Bounds3)::Tuple{Point3f,Float32}
    center = (b.p_min + b.p_max) / 2f0
    radius = inside(b, center) ? distance(center, b.p_max) : 0f0
    center, radius
end

function intersect(b::Bounds3, ray::AbstractRay)::Tuple{Bool,Float32,Float32}
    t0, t1 = 0f0, ray.t_max
    @inbounds for i in 1:3
        # Update interval for i-th bbox slab.
        inv_ray_dir = 1f0 / ray.d[i]
        t_near = (b.p_min[i] - ray.o[i]) * inv_ray_dir
        t_far = (b.p_max[i] - ray.o[i]) * inv_ray_dir
        if t_near > t_far
            t_near, t_far = t_far, t_near
        end

        t0 = t_near > t0 ? t_near : t0
        t1 = t_far < t1 ? t_far : t1
        t0 > t1 && return false, 0f0, 0f0
    end
    true, t0, t1
end

@inline function is_dir_negative(dir::Vec3f)
    @inbounds Point3{UInt8}(
        dir[1] < 0 ? 2 : 1,
        dir[2] < 0 ? 2 : 1,
        dir[3] < 0 ? 2 : 1,
    )
end

"""
dir_is_negative: 1 -- false, 2 -- true
"""
function intersect_p(
    b::Bounds3, ray::AbstractRay,
    inv_dir::Vec3f, dir_is_negative::Point3{UInt8},
)::Bool
    tx_min = (b[dir_is_negative[1]][1] - ray.o[1]) * inv_dir[1]
    tx_max = (b[3-dir_is_negative[1]][1] - ray.o[1]) * inv_dir[1]
    ty_min = (b[dir_is_negative[2]][2] - ray.o[2]) * inv_dir[2]
    ty_max = (b[3-dir_is_negative[2]][2] - ray.o[2]) * inv_dir[2]

    (tx_min > ty_max || ty_min > tx_max) && return false
    ty_min > tx_min && (tx_min = ty_min)
    ty_max > tx_max && (tx_max = ty_max)

    tz_min = (b[dir_is_negative[3]][3] - ray.o[3]) * inv_dir[3]
    tz_max = (b[3-dir_is_negative[3]][3] - ray.o[3]) * inv_dir[3]
    (tx_min > tz_max || tz_min > tx_max) && return false

    (tz_min > tx_min) && (tx_min = tz_min)
    (tz_max < tx_max) && (tx_max = tz_max)
    tx_min < ray.t_max && tx_max > 0
end
