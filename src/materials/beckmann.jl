# Ported from Blender Cycles v5.2: kernel/closure/bsdf_microfacet.h and util/math_fast.h.
# Copyright 2004 NVIDIA Corporation; 2008-2014 Larry Gritz;
# 2009-2014 Sony Pictures Imageworks Inc., et al.; 2011-2022 Blender Foundation.
# SPDX-License-Identifier: BSD-3-Clause. Modified: rewritten in Julia.
# See LICENSES/BSD-3-Clause.txt.

@inline function beckmann_erf(x::Float32)
    a = abs(x)
    a >= 12.3f0 && return copysign(1f0, x)
    b = 1f0 - (1f0 - a)
    r = muladd(muladd(muladd(muladd(muladd(muladd(0.0000430638f0, b, 0.0002765672f0),
        b, 0.0001520143f0), b, 0.0092705272f0), b, 0.0422820123f0), b, 0.0705230784f0), b, 1f0)
    s = r * r
    t = s * s
    u = t * t
    return copysign(1f0 - 1f0 / (u * u), x)
end

# Giles' inverse-erf approximation, including Cycles' endpoint clamp.
@inline function beckmann_erfinv(x::Float32)
    a = min(abs(x), 0.99999994f0)
    w = -log((1f0 - a) * (1f0 + a))
    if w < 5f0
        w -= 2.5f0
        p = 2.81022636f-8
        p = muladd(p, w, 3.43273939f-7)
        p = muladd(p, w, -3.5233877f-6)
        p = muladd(p, w, -4.39150654f-6)
        p = muladd(p, w, 0.00021858087f0)
        p = muladd(p, w, -0.00125372503f0)
        p = muladd(p, w, -0.00417768164f0)
        p = muladd(p, w, 0.246640727f0)
        p = muladd(p, w, 1.50140941f0)
    else
        w = sqrt(w) - 3f0
        p = -0.000200214257f0
        p = muladd(p, w, 0.000100950558f0)
        p = muladd(p, w, 0.00134934322f0)
        p = muladd(p, w, -0.00367342844f0)
        p = muladd(p, w, 0.00573950773f0)
        p = muladd(p, w, -0.0076224613f0)
        p = muladd(p, w, 0.00943887047f0)
        p = muladd(p, w, 1.00167406f0)
        p = muladd(p, w, 2.83297682f0)
    end
    return p * x
end

@inline function beckmann_d(h::Vec3f, alpha_x::Float32, alpha_y::Float32)
    z2 = h[3]^2
    z2 > 0f0 || return 0f0
    e = ((h[1] / alpha_x)^2 + (h[2] / alpha_y)^2) / z2
    return exp(-e) / (Float32(π) * alpha_x * alpha_y * z2^2)
end

@inline function beckmann_lambda(w::Vec3f, alpha_x::Float32, alpha_y::Float32)
    sqr_alpha_tan = ((alpha_x * w[1])^2 + (alpha_y * w[2])^2) / w[3]^2
    sqr_alpha_tan < 0.39f0 && return 0f0
    a = inv(sqrt(sqr_alpha_tan))
    return ((0.396f0 * a - 1.259f0) * a + 1f0) / ((2.181f0 * a + 3.535f0) * a)
end

# Heitz and d'Eon's visible-normal sampling, with Cycles' bounded Newton/bisection solve.
@inline function beckmann_sample_wm(w::Vec3f, u::Point2f, alpha_x::Float32, alpha_y::Float32)
    stretched = normalize(Vec3f(alpha_x * w[1], alpha_y * w[2], w[3]))
    cos_phi, sin_phi = 1f0, 0f0
    slope_x, slope_y = 0f0, 0f0
    if stretched[3] >= 0.99999f0
        # Sobol samples include zero. Keep that endpoint finite instead of
        # normalizing an infinite slope into NaNs; interior samples are unchanged.
        r = sqrt(-log(max(u[1], floatmin(Float32))))
        sin_angle, cos_angle = sincos(2f0 * Float32(π) * u[2])
        slope_x, slope_y = r * cos_angle, r * sin_angle
    else
        cos_theta = stretched[3]
        sin_theta = sqrt(max(1f0 - cos_theta^2, 0f0))
        tan_theta = sin_theta / cos_theta
        cot_theta = inv(tan_theta)
        erf_a = beckmann_erf(cot_theta)
        exp_a2 = exp(-cot_theta^2)
        cos_phi, sin_phi = stretched[1] / sin_theta, stretched[2] / sin_theta
        k = tan_theta * 0.56418958354f0
        y_approx = u[1] * (1f0 + erf_a + k * (1f0 - erf_a^2))
        y_exact = u[1] * (1f0 + erf_a + k * exp_a2)
        b = k > 0f0 ? (0.5f0 - sqrt(k * (k - y_approx + 1f0) + 0.25f0)) / k : y_approx - 1f0
        inverse_erf = beckmann_erfinv(b)
        lower_x, lower_y = -1f0, -y_exact
        upper_x = erf_a
        current_x = b
        current_y = 1f0 + b + k * exp(-inverse_erf^2) - y_exact
        for _ in 1:3
            abs(current_y) > 1f-6 || break
            if sign(lower_y) == sign(current_y)
                lower_x, lower_y = current_x, current_y
            else
                upper_x = current_x
            end
            newton = current_x - current_y / (1f0 - inverse_erf * tan_theta)
            current_x = lower_x <= newton <= upper_x ? newton : 0.5f0 * (lower_x + upper_x)
            inverse_erf = beckmann_erfinv(current_x)
            current_y = 1f0 + current_x + k * exp(-inverse_erf^2) - y_exact
        end
        slope_x, slope_y = inverse_erf, beckmann_erfinv(2f0 * u[2] - 1f0)
    end
    x = alpha_x * (cos_phi * slope_x - sin_phi * slope_y)
    y = alpha_y * (sin_phi * slope_x + cos_phi * slope_y)
    return normalize(Vec3f(-x, -y, 1f0))
end
