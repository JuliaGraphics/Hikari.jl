# Ported from Blender Cycles v5.2.1: intern/cycles/kernel/svm/closure.h, displace.h and
# kernel/closure/bsdf_sheen.h, bsdf_oren_nayar.h, copyright 2011-2022 Blender Foundation,
# SPDX-License-Identifier: Apache-2.0; and kernel/closure/bsdf_microfacet.h, bsdf_util.h, adapted
# by Cycles from Open Shading Language, copyright 2009-2010 Sony Pictures Imageworks Inc., et al.
# All Rights Reserved, copyright 2011-2022 Blender Foundation, SPDX-License-Identifier: BSD-3-Clause.
# Modified: rewritten in Julia; the thin film is evaluated per wavelength by the Airy sum. See
# LICENSES/Apache-2.0.txt and LICENSES/BSD-3-Clause.txt.

# ============================================================================
# Shader programs: the values a shader's BSDF nodes read, as Cycles' SVM computes them
# ============================================================================
#
# A shader graph's program computes every input and weight of its BSDF nodes like any other
# value on the stack, then holds the BSDF instructions after them (`compile_shader`). The
# interpreter runs the values (`run_values`); the BSDF nodes, typed in the material
# (`bsdf-nodes.jl`), read their inputs and set up their lobes (`lobes.jl`) from the helpers here,
# the way Cycles' `svm_node_closure_bsdf` does.

"""The kind of a BSDF node a `CLOSURE` instruction holds in its first mode byte."""
module ClosureKind
const DIFFUSE     = UInt8(1)
const GLOSSY      = UInt8(2)
const METALLIC    = UInt8(3)
const TRANSLUCENT = UInt8(4)
const GLASS       = UInt8(5)
const SHEEN       = UInt8(6)
const TRANSPARENT = UInt8(7)
const EMISSION    = UInt8(8)
const METALLIC_FILM = UInt8(9)
const GLASS_FILM = UInt8(10)
end

const CLOSURE_WEIGHT_CUTOFF = 1f-5
const THIN_FILM_THICKNESS_CUTOFF = 0.1f0

"""
What a shader program reads besides the texture inputs: where the hit is seen from, the
position's screen-space differentials (for bump), the shading tangent and the geometric normal,
both facing outwards, the handedness of the tangent frame the uv map makes (1 where the
bitangent `ns × dpdus` runs along increasing v, -1 where the uv map is mirrored), and where the
hit is in the camera's space and on the image, for texture coordinates.
"""
struct ShaderContext
    tfc::TextureFilterContext
    wo::Vec3f
    dpdx::Vec3f
    dpdy::Vec3f
    dpdus::Vec3f
    ng::Vec3f
    tangent_sign::Float32
    p_camera::Vec3f
    window::Point2f
end

@inline _max(v::Vec3f) = max(v[1], v[2], v[3])
@inline _pow3(v::Vec3f, e::Float32) = Vec3f(v[1]^e, v[2]^e, v[3]^e)

"""The specular layer's reflectance at normal incidence and IOR, as the Principled BSDF's `specular_ior_level` sets them."""
@inline function _specular_f0(ior::Float32, level::Float32)
    level = max(level, 0f0)
    f0 = _f0_from_ior(ior)
    return level == 0.5f0 ? f0 : f0 * 2f0 * level
end
@inline function _specular_eta(ior::Float32, level::Float32)
    max(level, 0f0) == 0.5f0 && return ior
    eta = _ior_from_f0(_specular_f0(ior, level))
    return ior < 1f0 ? 1f0 / eta : eta
end

"""Cycles' `closure_layering_weight`: what a layer of `albedo` leaves of `weight` for the layers under it."""
@inline function _layering(albedo::Vec3f, weight::Vec3f)
    ratio = Vec3f(ntuple(k -> weight[k] != 0f0 ? albedo[k] / weight[k] : 0f0, 3))
    return weight * clamp(1f0 - _max(ratio), 0f0, 1f0)
end
@inline _avg(v::Vec3f) = (v[1] + v[2] + v[3]) / 3f0
@inline _saturate3(v::Vec3f) = clamp.(v, 0f0, 1f0)
@inline _unit(n::Vec3f, fallback::Vec3f) = (l = norm(n); l > 1f-8 ? n / l : fallback)

"""Whether `wo` sees the back of the surface, as Cycles' `SD_BACKFACING`."""
@inline _backfacing(sc::ShaderContext) = dot(sc.wo, sc.ng) < 0f0

"""The geometric normal on the side `wo` is on: the frame Cycles' closures live in."""
@inline _facing_ng(sc::ShaderContext) = _backfacing(sc) ? -sc.ng : sc.ng

"""A lobe's normal, unit and on the side `wo` is on, as Cycles flips `N` with `Ng` for a backfacing hit."""
@inline function _closure_normal(n::Vec3f, sc::ShaderContext)
    nn = _unit(n, sc.tfc.n)
    return _backfacing(sc) ? -nn : nn
end

"""Cycles' `ensure_valid_specular_reflection`: `n` tilted towards `ng` until the mirror of `wo` leaves the surface."""
@inline function _valid_reflection_normal(ng::Vec3f, wo::Vec3f, n::Vec3f)
    r = 2f0 * dot(n, wo) * n - wo
    iz = dot(wo, ng)
    threshold = min(0.9f0 * iz, 0.01f0)
    dot(ng, r) >= threshold && return n
    x = _unit(n - dot(n, ng) * ng, n)
    ix = dot(wo, x)
    a = ix^2 + iz^2
    b = 2f0 * (a + iz * threshold)
    c = (threshold + iz)^2
    root = sqrt(max(b^2 - 4f0 * a * c, 0f0))
    nz2 = ix < 0f0 ? 0.25f0 * (b + root) / a : 0.25f0 * (b - root) / a
    return sqrt(max(1f0 - nz2, 0f0)) * x + sqrt(max(nz2, 0f0)) * ng
end

@inline _valid_normal(n::Vec3f, sc::ShaderContext) = _valid_reflection_normal(_facing_ng(sc), sc.wo, n)

"""A tangent for an anisotropic lobe: `t` in the plane of `n`, rotated around it by `rotation` turns."""
@inline function _anisotropic_tangent(t::Vec3f, n::Vec3f, rotation::Float32, fallback::Vec3f)
    t = _unit(t - dot(t, n) * n, _unit(fallback - dot(fallback, n) * n, coordinate_system(n)[1]))
    rotation == 0f0 && return t
    s, c = sincos(rotation * 2f0 * Float32(π))
    return c * t + s * cross(n, t)
end

@inline _f0_from_ior(ior::Float32) = ((ior - 1f0) / (ior + 1f0))^2
@inline function _ior_from_f0(f0::Float32)
    r = sqrt(clamp(f0, 0f0, 0.99f0))
    return (1f0 + r) / (1f0 - r)
end

"""The real dielectric Fresnel remapped to go from 0 at normal incidence to 1 at grazing, as Cycles' generalized Schlick with a negative exponent."""
@inline function _schlick_shape(cos_i::Float32, ior::Float32)
    r0 = _f0_from_ior(ior)
    return clamp((fresnel_dielectric(cos_i, ior) - r0) / max(1f0 - r0, 1f-6), 0f0, 1f0)
end

"""Cycles' `fresnel_dielectric_Fss`: the average single-scattering Fresnel of a dielectric."""
@inline _dielectric_fss(eta::Float32) =
    eta < 1f0 ? 0.997118f0 + eta * (0.1014f0 - eta * (0.965241f0 + eta * 0.130607f0)) : (eta - 1f0) / (4.08567f0 + 1.00071f0 * eta)

@inline function _f82_b(f0::Vec3f, tint::Vec3f)
    f = 6f0 / 7f0
    f5 = f^5
    return (f0 .+ (1f0 .- f0) .* f5) .* (7f0 / (f5 * f)) .* (1f0 .- tint)
end
@inline _f82_fss(f0::Vec3f, b::Vec3f) = f0 .+ (1f0 .- f0) ./ 21f0 .- b ./ 126f0

# ── Lookup-table albedos ────────────────────────────────────────────────────

@inline _ior_index(ior::Float32) = sqrt(abs((ior - 1f0) / (ior + 1f0)))

# ── Thin film ───────────────────────────────────────────────────────────────

"""
The reflectance at wavelength `lambda` (nm) of a substrate under a thin film, by the Airy sum of
the film's two interfaces, averaged over both polarizations. The substrate is a dielectric of
`eta3`, or with `k3 > 0` a conductor.

In `Float32` throughout (`complex_div`, `complex_sqrt`), as pbrt's complex arithmetic: Base's
complex division widens to `Complex{Float64}` and its square root rescales with `ldexp`, which made
this function a fifth of every Principled shading kernel, in `Float64` code that Apple GPUs cannot
compile at all.
"""
@noinline function _thin_film_reflectance(cos_i::Float32, thickness::Float32, eta2::Float32, eta3::Float32,
                                        k3::Float32, lambda::Float32, f82::Float32 = -1f0)
    if thickness < 1f0
        t = clamp(thickness, 0f0, 1f0)
        eta2 = 1f0 + (eta2 - 1f0) * t^2 * (3f0 - 2f0 * t)
    end
    sin2_i = max(1f0 - cos_i^2, 0f0)
    sin2_2 = sin2_i / eta2^2
    sin2_2 >= 1f0 && return 1f0
    cos_2 = sqrt(1f0 - sin2_2)
    n3 = Complex{Float32}(eta3, k3)
    cos_3 = complex_sqrt(Complex{Float32}(1f0, 0f0) - complex_div(sin2_i, n3 * n3))
    # The film's phase is imaginary, so its exponential is a rotation.
    shift = cis(4f0 * Float32(π) * eta2 * thickness * cos_2 / lambda)
    r = 0f0
    for polarization in (0x00, 0x01)
        r12 = polarization == 0x00 ? (cos_i - eta2 * cos_2) / (cos_i + eta2 * cos_2) :
                                     (eta2 * cos_i - cos_2) / (eta2 * cos_i + cos_2)
        r23 = polarization == 0x00 ? complex_div(eta2 * cos_2 - n3 * cos_3, eta2 * cos_2 + n3 * cos_3) :
                                     complex_div(n3 * cos_2 - eta2 * cos_3, n3 * cos_2 + eta2 * cos_3)
        if f82 >= 0f0
            # Cycles keeps the artist's F82 reflectance at the film/substrate interface;
            # the fitted complex IOR supplies its phase, rather than its magnitude.
            n = eta3 / eta2
            k2 = (k3 / eta2)^2
            f0 = ((n - 1f0)^2 + k2) / ((n + 1f0)^2 + k2)
            s = 6f0 / 7f0
            b = (f0 + (1f0 - f0) * s^5 - f82) * (7f0 / s^6)
            m = 1f0 - cos_2
            R = clamp(f0 + (1f0 - f0) * m^5 - b * cos_2 * m^6, 0f0, 1f0)
            magnitude = abs(r23)
            r23 = magnitude > 0f0 ? r23 * (sqrt(R) / magnitude) : Complex{Float32}(sqrt(R), 0f0)
        end
        rr = complex_div(r12 + r23 * shift, 1f0 + r12 * r23 * shift)
        r += abs2(rr)
    end
    return clamp(0.5f0 * r, 0f0, 1f0)
end

"""
A conductor's IOR and extinction at one wavelength from its reflectance at normal incidence and
at about 82°, as Cycles estimates them for a thin film (Gulbrandsen's artist-friendly Fresnel).
"""
@inline function _conductor_nk(f0::Float32, b::Float32)
    r = min(f0, 0.999f0)
    m = 1f0 - 1f0 / 7f0
    m5 = m^5
    g = clamp(f0 + (1f0 - f0) * m5 - b * (1f0 / 7f0) * m5 * m, 0f0, 1f0)
    sqrt_r = sqrt(r)
    n = (1f0 - g) * (1f0 + sqrt_r) / (1f0 - sqrt_r) + g * (1f0 - r) / (1f0 + r)
    k = sqrt(max((r * (n + 1f0)^2 - (n - 1f0)^2) / (1f0 - r), 0f0))
    return n, k
end

@inline function _f82_b_channel(f0::Float32, tint::Float32)
    f = 6f0 / 7f0
    f5 = f^5
    return (f0 + (1f0 - f0) * f5) * (7f0 / (f5 * f)) * (1f0 - tint)
end

# ── Sheen and Oren-Nayar ────────────────────────────────────────────────────

@inline function _oren_nayar_g(cos_theta::Float32)
    cos_theta < 1f-6 && return (Float32(π) / 2f0 - 2f0 / 3f0) - cos_theta
    sin_theta = sqrt(max(1f0 - cos_theta^2, 0f0))
    theta = acos(clamp(cos_theta, -1f0, 1f0))
    return sin_theta * (theta - 2f0 / 3f0 - sin_theta * cos_theta) +
           2f0 / 3f0 * (sin_theta / cos_theta) * (1f0 - sin_theta^2 * sin_theta)
end

"""
The alphas of a GGX lobe of `roughness` stretched by `anisotropy` along its tangent, as Cycles:
the tangent input where there is one, else the shading tangent, rotated by `rotation` turns.
"""
@inline function _anisotropic_alphas(roughness::Float32, anisotropy::Float32, tangent::Vec3f, n::Vec3f,
                                     rotation::Float32, sc::ShaderContext)
    alpha = roughness^2
    anisotropy == 0f0 && return alpha, alpha, Vec3f(0)
    aspect = sqrt(1f0 - abs(anisotropy) * 0.9f0)
    t = _anisotropic_tangent(tangent, n, rotation, sc.dpdus)
    anisotropy < 0f0 && (t = cross(n, t))
    return alpha / aspect, alpha * aspect, t
end

"""The inputs of a Principled BSDF, read off its instructions before its lobes are set up."""
struct PrincipledInputs
    weight::Float32
    base::Vec3f
    metallic::Float32
    roughness::Float32
    ior::Float32
    alpha::Float32
    n::Vec3f
    diffuse_roughness::Float32
    subsurface::Float32
    specular_level::Float32
    specular_tint::Vec3f
    anisotropic::Float32
    rotation::Float32
    tangent::Vec3f
    transmission::Float32
    dispersion::Float32
    abbe::Float32
    coat_weight::Float32
    coat_roughness::Float32
    coat_ior::Float32
    coat_tint::Vec3f
    coat_n::Vec3f
    sheen_weight::Float32
    sheen_roughness::Float32
    sheen_tint::Vec3f
    emission::Vec3f
    film_thickness::Float32
    film_ior::Float32
    subsurface_anisotropy::Float32
    thin_wall::Float32
    subsurface_radius::Vec3f
    subsurface_scale::Float32
end

# ── Bump ────────────────────────────────────────────────────────────────────

"""
Blender's Bump node, as Cycles' `svm_node_set_bump`: the height at the hit and at the hit moved
by `filter_width` of each screen-space differential, turned into a normal.
"""
@inline function _bump(s, ins::NodeInstr, sc::ShaderContext)
    h_c, h_x, h_y = _f(s, ins, 1), _f(s, ins, 2), _f(s, ins, 3)
    strength = clamp(_f(s, ins, 4), 0f0, 1f0)
    scale = ins.mode[1] == 0x01 ? -_f(s, ins, 5) : _f(s, ins, 5)
    normal_in = _unit(_v(s, ins, 6), sc.tfc.n)
    filter_width = _f(s, ins, 7)
    dpdx, dpdy = _bump_differentials(sc)

    rx = cross(dpdy, normal_in)
    ry = cross(normal_in, dpdx)
    det = dot(dpdx, rx)
    surfgrad = (h_x - h_c) * rx + (h_y - h_c) * ry
    normal_out = filter_width * abs(det) * normal_in - scale * sign(det) * surfgrad
    l = norm(normal_out)
    l > 0f0 || return normal_in
    return _unit(strength * (normal_out / l) + (1f0 - strength) * normal_in, normal_in)
end

"""The differentials a bump shifts the hit by; along the shading tangents when the hit has none."""
@inline function _bump_differentials(sc::ShaderContext)
    dot(sc.dpdx, sc.dpdx) > 0f0 && dot(sc.dpdy, sc.dpdy) > 0f0 && return sc.dpdx, sc.dpdy
    t = _unit(sc.dpdus - dot(sc.dpdus, sc.tfc.n) * sc.tfc.n, coordinate_system(sc.tfc.n)[1])
    return BUMP_DEFAULT_DELTA * t, BUMP_DEFAULT_DELTA * cross(sc.tfc.n, t)
end

# ── Normal map ──────────────────────────────────────────────────────────────

const NORMAL_MAP_TANGENT = 0x00
const NORMAL_MAP_WORLD = 0x02
const NORMAL_MAP_BLENDER_WORLD = 0x04

"""
Blender's Normal Map node, as Cycles' `svm_node_normal_map`: the colour's normal in the frame of
the shading tangent, the bitangent of the uv map's handedness and the shading normal, or in
world space. Strength scales the tangent space normal, and interpolates towards the shading
normal in world space.
"""
@inline function _normal_map(s, ins::NodeInstr, sc::ShaderContext)
    c = 2f0 .* (_v(s, ins, 1) .- 0.5f0)
    c = ins.mode[2] == 0x01 ? Vec3f(c[1], -c[2], c[3]) : c
    strength = _f(s, ins, 2)
    n = sc.tfc.n
    if ins.mode[1] == NORMAL_MAP_TANGENT
        t = _unit(sc.dpdus - dot(sc.dpdus, n) * n, coordinate_system(n)[1])
        b = sc.tangent_sign * cross(n, t)
        tilted = Vec3f(c[1] * strength, c[2] * strength, 1f0 + (c[3] - 1f0) * clamp(strength, 0f0, 1f0))
        return _unit(tilted[1] * t + tilted[2] * b + tilted[3] * n, n)
    end
    c = ins.mode[1] == NORMAL_MAP_BLENDER_WORLD ? Vec3f(c[1], -c[2], -c[3]) : c
    mapped = _unit(c, n)
    strength == 1f0 && return mapped
    return _unit(n + (mapped - n) * max(strength, 0f0), n)
end

# ── Mesh attributes ─────────────────────────────────────────────────────────

"""
The attribute an `ATTRIBUTE` instruction reads at the hit, or at the hit moved by a filter width
of a screen-space differential for a bump's `ATTRIBUTE_DX` and `ATTRIBUTE_DY`. Unbound, it is 0.
"""
@inline function _attribute(textures, ins::NodeInstr, sc::ShaderContext)
    slot, idx = reinterpret(Int32, ins.k[9]), reinterpret(Int32, ins.k[10])
    slot > Int32(0) || return Vec3f(0)
    tfc = sc.tfc
    fw = ins.k[1]
    duv = ins.op == NodeOp.ATTRIBUTE_DX ? Vec2f(fw * tfc.dudx, fw * tfc.dvdx) :
          ins.op == NodeOp.ATTRIBUTE_DY ? Vec2f(fw * tfc.dudy, fw * tfc.dvdy) : Vec2f(0)
    return Raycore.with_texture(_corner_attribute, textures, slot, idx, tfc.face_idx, tfc.bary, duv)
end

"""
An attribute stored per triangle as its three corner values and the corners' uv: the corners
mixed by the barycentrics `bary`, moved by as much as the uv moves by `duv`.
"""
@propagate_inbounds function _corner_attribute(corners::AbstractVector{Vec3f}, face::UInt32, bary, duv::Vec2f)
    triangles = length(corners) ÷ 6
    (face >= UInt32(1) && triangles > 0) || return Vec3f(0)
    base = 6 * (min(Int(face), triangles) - 1)
    a1, a2, a3 = corners[base + 1], corners[base + 2], corners[base + 3]
    value = bary[1] * a1 + bary[2] * a2 + bary[3] * a3
    duv == Vec2f(0) && return value
    t1, t2, t3 = corners[base + 4], corners[base + 5], corners[base + 6]
    e2, e3 = Vec2f(t2[1] - t1[1], t2[2] - t1[2]), Vec2f(t3[1] - t1[1], t3[2] - t1[2])
    det = e2[1] * e3[2] - e3[1] * e2[2]
    abs(det) > 1f-12 || return value
    db2 = (duv[1] * e3[2] - e3[1] * duv[2]) / det
    db3 = (e2[1] * duv[2] - duv[1] * e2[2]) / det
    return value + db2 * (a2 - a1) + db3 * (a3 - a1)
end
@propagate_inbounds _corner_attribute(_, face::UInt32, bary, duv::Vec2f) = Vec3f(0)

@inline function _shifted_leaf(op::UInt8, s, ins::NodeInstr, sc::ShaderContext)
    fw = ins.k[1]
    dpdx, dpdy = _bump_differentials(sc)
    tfc = sc.tfc
    op == NodeOp.POSITION_DX && return Vec3f(tfc.p) + fw * dpdx
    op == NodeOp.POSITION_DY && return Vec3f(tfc.p) + fw * dpdy
    op == NodeOp.UV_DX && return Vec3f(tfc.uv[1] + fw * tfc.dudx, tfc.uv[2] + fw * tfc.dvdx, 0f0)
    return Vec3f(tfc.uv[1] + fw * tfc.dudy, tfc.uv[2] + fw * tfc.dvdy, 0f0)
end

"""
Run a shader program's value instructions at a hit: the stack they leave, and the position of
its first BSDF instruction, which a program places after every value (`compile_shader`).
"""
@inline function run_values(prog::AbstractVector{NodeInstr{F}}, textures, sc::ShaderContext) where {F}
    s = _new_stack(Val(_stack_size(F)))
    i = Int32(1)
    n = length(prog) % Int32
    @inbounds while i <= n
        ins = prog[i]
        op = ins.op
        (op == NodeOp.RESULT || op == NodeOp.CLOSURE || op == NodeOp.PRINCIPLED) && return s, i
        skip = Int32(0)
        if _has(ins, NodeFeature.BUMP) && op == NodeOp.BUMP
            s = _store(s, ins, _bump(s, ins, sc))
        elseif op == NodeOp.NORMAL_MAP
            s = _store(s, ins, _normal_map(s, ins, sc))
        elseif _has(ins, NodeFeature.ATTRIBUTE) && NodeOp.ATTRIBUTE <= op <= NodeOp.ATTRIBUTE_DY
            a = _attribute(textures, ins, sc)
            s = _store(s, ins, a, _splat((a[1] + a[2] + a[3]) / 3f0))
        elseif op == NodeOp.CAMERA
            s = _store(s, ins, sc.p_camera)
        elseif op == NodeOp.WINDOW
            s = _store(s, ins, Vec3f(sc.window[1], sc.window[2], 0f0))
        elseif op == NodeOp.REFLECTION
            facing = _closure_normal(sc.tfc.n, sc)
            s = _store(s, ins, 2f0 * dot(facing, sc.wo) * facing - sc.wo)
        elseif _has(ins, NodeFeature.BUMP) && NodeOp.POSITION_DX <= op <= NodeOp.UV_DY
            s = _store(s, ins, _shifted_leaf(op, s, ins, sc))
        else
            s, skip = _exec(s, prog, i, ins, textures, sc.tfc)
        end
        i += Int32(1) + skip
    end
    return s, i
end

