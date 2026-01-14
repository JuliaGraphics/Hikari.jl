# ============================================================================
# MixMaterial - Stochastically blends between two materials
# ============================================================================
# Port of pbrt-v4's MixMaterial using stochastic material selection
#
# The material uses a mixing amount texture to blend between two materials:
# - amount = 0: always select material 1
# - amount = 1: always select material 2
# - 0 < amount < 1: stochastically select based on deterministic hash
#
# Reference: pbrt-v4 src/pbrt/materials.h MixMaterial
#
# IMPORTANT: MixMaterial is resolved at intersection time, not at BSDF evaluation.
# This means the integrator's intersection kernel must call choose_material()
# before creating material evaluation work items.

"""
    MixMaterial{M1, M2, AmountTex}

A material that stochastically blends between two sub-materials based on a mixing amount.

Following pbrt-v4, the material selection is deterministic based on:
- The intersection position and viewing direction
- A hash function to generate deterministic randomness
- The `amount` texture value at the hit point

# Fields
- `material1`: First material (selected when amount → 0)
- `material2`: Second material (selected when amount → 1)
- `amount`: Texture controlling the blend ratio (0 = material1, 1 = material2)
- `material1_idx`: Index of material1 in the materials tuple
- `material2_idx`: Index of material2 in the materials tuple

# Usage
MixMaterial is resolved at intersection time before material evaluation.
The integrator should call `choose_material()` to get the actual material index
to use for the hit point, then proceed with normal material evaluation.
"""
struct MixMaterial{M1<:Material, M2<:Material, AmountTex} <: Material
    material1::M1
    material2::M2
    amount::AmountTex
    # Store material indices for use after resolution
    material1_idx::MaterialIndex
    material2_idx::MaterialIndex
end

# Constructor with material tuple indices
function MixMaterial(
    material1::M1,
    material2::M2,
    amount::Texture,
    material1_idx::MaterialIndex,
    material2_idx::MaterialIndex
) where {M1<:Material, M2<:Material}
    MixMaterial{M1, M2, typeof(amount)}(material1, material2, amount, material1_idx, material2_idx)
end

"""
    MixMaterial(; materials, amount, material_indices)

Create a MixMaterial with keyword arguments.

# Arguments
- `materials`: Tuple of two materials (material1, material2)
- `amount`: Mixing amount (0-1 scalar, texture, or image path)
- `material_indices`: Tuple of MaterialIndex for each material

# Examples
```julia
# Simple 50-50 blend
MixMaterial(
    materials=(gold_material, red_diffuse),
    amount=0.5,
    material_indices=(gold_idx, diffuse_idx)
)

# Texture-based blend (e.g., mask texture)
MixMaterial(
    materials=(gold_material, red_diffuse),
    amount=mask_texture,
    material_indices=(gold_idx, diffuse_idx)
)
```
"""
function MixMaterial(;
    materials::Tuple{<:Material, <:Material},
    amount=0.5f0,
    material_indices::Tuple{MaterialIndex, MaterialIndex}
)
    MixMaterial(
        materials[1],
        materials[2],
        _to_texture(amount),
        material_indices[1],
        material_indices[2]
    )
end

# MixMaterial is not directly emissive (emission comes from chosen sub-material)
is_emissive(::MixMaterial) = false

# ============================================================================
# Hash function for deterministic material selection
# ============================================================================

"""
    mix_hash_float(p::Point3f, wo::Vec3f, idx1::MaterialIndex, idx2::MaterialIndex) -> Float32

Generate a deterministic pseudo-random float in [0, 1) for material selection.
Uses a simple but effective hash function based on pbrt-v4's HashFloat.

The hash is deterministic: same position, direction, and materials always
produce the same result, ensuring consistent rendering across samples.
"""
@propagate_inbounds function mix_hash_float(
    p::Point3f, wo::Vec3f,
    idx1::MaterialIndex, idx2::MaterialIndex
)::Float32
    # MurmurHash-inspired mixing
    # We hash the position, direction, and material indices together

    # Convert to bits and combine
    h = UInt64(0)

    # Mix position (most important for spatial variation)
    h = xor(h, reinterpret(UInt32, p[1]))
    h = h * UInt64(0xcc9e2d51)
    h = xor(h, reinterpret(UInt32, p[2]) << 4)
    h = h * UInt64(0x1b873593)
    h = xor(h, reinterpret(UInt32, p[3]) << 8)

    # Mix direction
    h = xor(h, reinterpret(UInt32, wo[1]) << 16)
    h = h * UInt64(0xcc9e2d51)
    h = xor(h, reinterpret(UInt32, wo[2]))
    h = h * UInt64(0x1b873593)
    h = xor(h, reinterpret(UInt32, wo[3]) << 12)

    # Mix material indices
    h = xor(h, UInt64(idx1.material_type) << 24)
    h = xor(h, UInt64(idx1.material_idx))
    h = h * UInt64(0xcc9e2d51)
    h = xor(h, UInt64(idx2.material_type) << 28)
    h = xor(h, UInt64(idx2.material_idx) << 4)
    h = h * UInt64(0x1b873593)

    # Final mixing (from pbrt-v4's MixBits)
    h = xor(h, h >> 31)
    h = h * UInt64(0x7fb5d329728ea185)
    h = xor(h, h >> 27)
    h = h * UInt64(0x81dadef4bc2dd44d)
    h = xor(h, h >> 33)

    # Convert to float in [0, 1)
    # 0x1p-32 = 2^(-32) ≈ 2.328306e-10
    return Float32(UInt32(h & 0xFFFFFFFF)) * Float32(exp2(-32))
end

# ============================================================================
# Material Selection
# ============================================================================

"""
    choose_material(mix::MixMaterial, textures, p::Point3f, wo::Vec3f, uv::Point2f) -> MaterialIndex

Choose which sub-material to use at the given hit point.
Returns the MaterialIndex of the chosen material.

Following pbrt-v4's ChooseMaterial:
1. Evaluate the amount texture at (uv)
2. If amount ≤ 0, return material1
3. If amount ≥ 1, return material2
4. Otherwise, use deterministic hash to stochastically select

This function is called at intersection time, before material evaluation.
"""
@propagate_inbounds function choose_material(
    mix::MixMaterial, textures,
    p::Point3f, wo::Vec3f, uv::Point2f
)::MaterialIndex
    amt = eval_tex(textures, mix.amount, uv)

    # Early exit for boundary cases
    if amt <= 0f0
        return mix.material1_idx
    end
    if amt >= 1f0
        return mix.material2_idx
    end

    # Stochastic selection using deterministic hash
    u = mix_hash_float(p, wo, mix.material1_idx, mix.material2_idx)

    # Select material: if amount < hash, use material1, else material2
    # This gives material1 probability (1 - amount) and material2 probability (amount)
    return amt < u ? mix.material1_idx : mix.material2_idx
end

"""
    is_mix_material(mat) -> Bool

Check if a material is a MixMaterial.
"""
is_mix_material(::Material) = false
is_mix_material(::MixMaterial) = true

"""
    is_mix_material_dispatch(materials, idx::MaterialIndex) -> Bool

Type-stable dispatch to check if a material is MixMaterial.
"""
@propagate_inbounds @generated function is_mix_material_dispatch(
    materials::NTuple{N,Any}, idx::MaterialIndex
) where {N}
    branches = [quote
         if idx.material_type === UInt8($i)
            @inbounds return is_mix_material(materials[$i][idx.material_idx])
        end
    end for i in 1:N]

    quote
        $(branches...)
        return false
    end
end

"""
    choose_material_dispatch(materials, textures, idx::MaterialIndex, p, wo, uv) -> MaterialIndex

Type-stable dispatch for choosing material from MixMaterial.
If the material is not MixMaterial, returns the input index unchanged.
"""
@propagate_inbounds @generated function choose_material_dispatch(
    materials::NTuple{N,Any}, textures,
    idx::MaterialIndex,
    p::Point3f, wo::Vec3f, uv::Point2f
) where {N}
    branches = [quote
         if idx.material_type === UInt8($i)
            mat = @inbounds materials[$i][idx.material_idx]
            return is_mix_material(mat) ? choose_material(mat, textures, p, wo, uv) : idx
        end
    end for i in 1:N]

    quote
        $(branches...)
        return idx
    end
end

"""
    resolve_mix_material(materials, textures, idx::MaterialIndex, p, wo, uv) -> MaterialIndex

Resolve any MixMaterial chain to get the final material index.
Handles nested MixMaterials by iterating until a non-mix material is found.

This should be called at intersection time before creating material work items.
"""
@propagate_inbounds function resolve_mix_material(
    materials, textures,
    idx::MaterialIndex,
    p::Point3f, wo::Vec3f, uv::Point2f
)::MaterialIndex
    # Iterate to handle nested MixMaterials (up to 8 levels to prevent infinite loops)
    current_idx = idx
    for _ in 1:8
        if !is_mix_material_dispatch(materials, current_idx)
            return current_idx
        end
        current_idx = choose_material_dispatch(materials, textures, current_idx, p, wo, uv)
    end
    return current_idx
end

# NOTE: MixMaterial does NOT implement spectral evaluation functions.
# Following pbrt-v4, MixMaterial is always resolved at intersection time
# to a concrete material before any BSDF evaluation occurs.
# See pbrt-v4 materials.h line 339-344: GetBxDF() is LOG_FATAL if called.

"""Type alias: `Mix` is the same as `MixMaterial`"""
const Mix = MixMaterial
