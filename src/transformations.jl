struct Transformation
    m::Mat4f
    inv_m::Mat4f
end

Transformation() = Transformation(Mat4f(I), Mat4f(I))
Transformation(m::Mat4f) = Transformation(m, inv(m))
is_identity(t::Transformation) = t.m == I && t.inv_m == I
function Base.transpose(t::Transformation)
    Transformation(transpose(t.m), transpose(t.inv_m))
end
Base.inv(t::Transformation) = Transformation(t.inv_m, t.m)

function Base.:(==)(t1::Transformation, t2::Transformation)
    all(t1.m .== t2.m) && all(t1.inv_m .== t2.inv_m)
end
function Base.:≈(t1::Transformation, t2::Transformation)
    all(t1.m .≈ t2.m) && all(t1.inv_m .≈ t2.inv_m)
end
function Base.:*(t1::Transformation, t2::Transformation)
    Transformation(t1.m * t2.m, t1.inv_m * t2.inv_m)
end

function translate(δ::Vec3f)
    m = transpose(Mat4f(
        1, 0, 0, δ[1],
        0, 1, 0, δ[2],
        0, 0, 1, δ[3],
        0, 0, 0, 1,
    ))
    m_inv = transpose(Mat4f(
        1, 0, 0, -δ[1],
        0, 1, 0, -δ[2],
        0, 0, 1, -δ[3],
        0, 0, 0, 1,
    ))
    Transformation(m, m_inv)
end

function scale(x::Number, y::Number, z::Number)
    m = transpose(Mat4f(
        x, 0, 0, 0,
        0, y, 0, 0,
        0, 0, z, 0,
        0, 0, 0, 1,
    ))
    m_inv = transpose(Mat4f(
        1 / x, 0, 0, 0,
        0, 1 / y, 0, 0,
        0, 0, 1 / z, 0,
        0, 0, 0, 1,
    ))
    Transformation(m, m_inv)
end

function rotate_x(θ::Float32)
    sin_θ = sin(deg2rad(θ))
    cos_θ = cos(deg2rad(θ))
    m = transpose(Mat4f(
        1, 0, 0, 0,
        0, cos_θ, -sin_θ, 0,
        0, sin_θ, cos_θ, 0,
        0, 0, 0, 1,
    ))
    Transformation(m, transpose(m))
end

function rotate_y(θ::Float32)
    sin_θ = sin(deg2rad(θ))
    cos_θ = cos(deg2rad(θ))
    m = transpose(Mat4f(
        cos_θ, 0, sin_θ, 0,
        0, 1, 0, 0,
        -sin_θ, 0, cos_θ, 0,
        0, 0, 0, 1,
    ))
    Transformation(m, transpose(m))
end

function rotate_z(θ::Float32)
    sin_θ = sin(deg2rad(θ))
    cos_θ = cos(deg2rad(θ))
    m = transpose(Mat4f(
        cos_θ, -sin_θ, 0, 0,
        sin_θ, cos_θ, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1,
    ))
    Transformation(m, transpose(m))
end

function rotate(θ::Float32, axis::Vec3f)
    a = normalize(axis)
    sin_θ = sin(deg2rad(θ))
    cos_θ = cos(deg2rad(θ))
    m = transpose(Mat4f(
        a[1] * a[1] + (1 - a[1] * a[1]) * cos_θ, a[1] * a[2] * (1 - cos_θ) - a[3] * sin_θ, a[1] * a[3] * (1 - cos_θ) + a[2] * sin_θ, 0,
        a[1] * a[2] * (1 - cos_θ) + a[3] * sin_θ, a[2] * a[2] + (1 - a[2] * a[2]) * cos_θ, a[2] * a[3] * (1 - cos_θ) - a[1] * sin_θ, 0,
        a[1] * a[3] * (1 - cos_θ) - a[2] * sin_θ, a[2] * a[3] * (1 - cos_θ) + a[1] * sin_θ, a[3] * a[3] + (1 - a[3] * a[3]) * cos_θ, 0,
        0, 0, 0, 1,
    ))
    Transformation(m, transpose(m))
end

function look_at(position::Point3f, target::Point3f, up::Vec3f)
    z_axis = normalize((position - target))
    x_axis = normalize((up × z_axis))
    y_axis = z_axis × x_axis

    m = transpose(Mat4f(
        x_axis[1], y_axis[1], z_axis[1], 0,
        x_axis[2], y_axis[2], z_axis[2], 0,
        x_axis[3], y_axis[3], z_axis[3], 0,
        0, 0, 0, 1,
    ))
    translate(Vec3f(position)) * Transformation(m, transpose(m))
end

function perspective(fov::Float32, near::Float32, far::Float32)
    # Projective divide for perspective projection.
    p = Mat4f(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, far / (far - near), -far * near / (far - near),
        0, 0, 1, 0,
    )
    # Scale canonical perspective view to specified field of view.
    inv_tan = 1f0 / tan(deg2rad(fov) / 2f0)
    scale(inv_tan, inv_tan, 1f0) * Transformation(p)
end

function (t::Transformation)(p::Point3f)::Point3f
    ph = Point4f(p..., 1f0)
    pt = t.m * ph
    pr = Point3f(pt[Vec(1, 2, 3)])
    pt[4] == 1 && return pr
    pr ./ pt[4]
end
(t::Transformation)(v::Vec3f)::Vec3f = t.m[Vec(1, 2, 3), Vec(1, 2, 3)] * v
(t::Transformation)(n::Normal3f)::Normal3f = transpose(t.inv_m[Vec(1, 2, 3), Vec(1, 2, 3)]) * n
function (t::Transformation)(b::Bounds3)
    mapreduce(i -> Bounds3(t(corner(b, i))), ∪, 1:8)
end
(t::Transformation)(r::Ray) = Ray(t(r.o), t(r.d), r.t_max, r.time)
function (t::Transformation)(rd::RayDifferentials)
    RayDifferentials(
        t(rd.o), t(rd.d), rd.t_max, rd.time,
        rd.has_differentials,
        t(rd.rx_origin),
        t(rd.ry_origin),
        t(rd.rx_direction),
        t(rd.ry_direction),
    )
end

function has_scale(t::Transformation)
    a = norm(t(Vec3f(1, 0, 0)))
    b = norm(t(Vec3f(0, 1, 0)))
    c = norm(t(Vec3f(0, 0, 1)))
    a ≉ 1 || b ≉ 1 || c ≉ 1
end

function swaps_handedness(t::Transformation)
    det(t.m[Vec(1, 2, 3), Vec(1, 2, 3)]) < 0
end

struct Quaternion
    v::Vec3f
    w::Float32
end

Quaternion() = Quaternion(Vec3f(0f0), 1f0)
function Quaternion(t::Transformation)
    trace = tr(t.m[Vec(1, 2, 3), Vec(1, 2, 3)])
    if trace > 0f0
        # Compute w from matrix trace, then xyz.
        # 4w^2 = m[0, 0] + m[1, 1] + m[2, 2] + m[3, 3] => (m[3, 3] == 1)
        s = sqrt(trace + 1f0)
        w = s / 2f0
        s = 0.5f0 / s
        v = Vec3f(
            (t.m[3, 2] - t.m[2, 3]) * s,
            (t.m[1, 3] - t.m[3, 1]) * s,
            (t.m[2, 1] - t.m[1, 2]) * s,
        )
    else
        # Compute largest x, y or z, then remaining components.
        nxt = Vec3(1, 2, 0)
        i = 0
        (t.m[2, 2] > t.m[1, 1]) && (i = 1)
        (t.m[3, 3] > t.m[i, i]) && (i = 2)
        j = nxt[i]
        k = nxt[j]

        q = Vector{Float32}(undef, 3)
        s = sqrt((t.m[i, i] - (t.m[j, j] + t.m[k, k])) + 1f0)
        q[i] = s * 0.5f0
        (s != 0f0) && (s = 0.5f0 / s)
        q[j] = (t.m[j, i] + t.m[i, j]) * s
        q[k] = (t.m[k, i] + t.m[i, k]) * s
        w = (t.m[k, j] - t.m[j, k]) * s
        v = Vec3f(q)
    end
    Quaternion(v, w)
end

Base.:+(q1::Quaternion, q2::Quaternion) = Quaternion(q1.v .+ q2.v, q1.w + q2.w)
Base.:-(q1::Quaternion, q2::Quaternion) = Quaternion(q1.v .- q2.v, q1.w - q2.w)
Base.:/(q::Quaternion, f::Float32) = Quaternion(q.v ./ f, q.w / f)
Base.:*(q::Quaternion, f::Float32) = Quaternion(q.v .* f, q.w * f)
LinearAlgebra.dot(q1::Quaternion, q2::Quaternion) = q1.v ⋅ q2.v + q1.w * q2.w
LinearAlgebra.normalize(q::Quaternion) = q / sqrt(q ⋅ q)

function Transformation(q::Quaternion)
    xx = q.v[1] * q.v[1]
    yy = q.v[2] * q.v[2]
    zz = q.v[3] * q.v[3]

    xy = q.v[1] * q.v[2]
    xz = q.v[1] * q.v[3]
    yz = q.v[2] * q.v[3]

    wx = q.w * q.v[1]
    wy = q.w * q.v[2]
    wz = q.w * q.v[3]

    m = Mat4f(
        1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 0,
        2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 0,
        2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy), 0,
        0, 0, 0, 1,
    )
    Transformation(m, transpose(m))
end

function slerp(q1::Quaternion, q2::Quaternion, t::Float32)
    cos_θ = q1 ⋅ q2
    cos_θ > 0.9995f0 && return normalize((1 - t) * q1 + t * q2)

    θ = acos(cos_θ)
    θ_p = θ * t
    q_perp = normalize(q2 - q1 * cos_θ)
    q1 * cos(θ_p) + q_perp * sin(θ_p)
end
