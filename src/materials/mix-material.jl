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
    MixMaterial

A material that stochastically blends between two sub-materials based on a mixing amount.

Following pbrt-v4, the material selection is deterministic based on:
- The intersection position and viewing direction
- A hash function to generate deterministic randomness
- The `amount` texture value at the hit point

# Fields
- `amount`: blend ratio (0 = material1, 1 = material2)
- `material1_idx`: scene-materials key of the first sub-material
- `material2_idx`: scene-materials key of the second sub-material

# Usage
MixMaterial is resolved at intersection time before material evaluation.
The integrator should call `choose_material()` to get the actual material index
to use for the hit point, then proceed with normal material evaluation.
"""
# NON-PARAMETRIC, and that matters more here than anywhere else: the
# MixMaterial closest-hit shader has to dispatch over EVERY material type in
# the scene (see rt-pipeline.jl `T <: MixMaterial`), so it inlines the whole
# shading system. One MixMaterial type per distinct sub-material PAIR therefore
# cost a full copy of that shader each — the two Crown mixes were the single
# largest item in its compile time.
#
# pbrt-v4 has the same shape: `MixMaterial` holds `Material materials[2]`,
# where `Material` is a TaggedPointer into a heterogeneous pool. `SetKey` is
# our TaggedPointer, so the sub-materials are referenced, not embedded. The
# consequence is that a MixMaterial can only be built against a scene, since
# that is what hands out the keys.
struct MixMaterial{AmountT} <: Material
    amount::AmountT
    material1_idx::SetKey
    material2_idx::SetKey
end

"""
    MixMaterialSpec(material1, material2, amount)

Host-side form of a [`MixMaterial`](@ref), holding the two sub-materials
themselves rather than their keys. A `Scene` hands out keys only at push time,
and materials are routinely built before any scene exists (RayMakie constructs
them inside a plot's argument-conversion node), so the mix carries its
sub-materials until then. `resolve_material` turns it into the device form.
"""
struct MixMaterialSpec{M1 <: Material, M2 <: Material, AmountT} <: Material
    material1::M1
    material2::M2
    amount::AmountT
end

is_emissive(::MixMaterialSpec) = false

"""
    MixMaterial(scene; materials, amount)

Create a MixMaterial that blends between two sub-materials, pushing both into
`scene`'s material set to obtain their keys.

# Arguments
- `materials`: Tuple of two materials (material1, material2)
- `amount`: Mixing amount (0-1 scalar or texture handle). 0 = material1, 1 = material2.

# Examples
```julia
MixMaterial(scene; materials=(gold, diffuse), amount=0.5)
MixMaterial(scene; materials=(gold, diffuse), amount=mask_handle)
```
"""
function MixMaterial(;
    materials::Tuple{<:Material, <:Material},
    amount=0.5f0,
)
    return MixMaterialSpec(materials[1], materials[2], matparam(amount))
end

"""
    resolve_material(scene, mat) -> mat

Turn any host-only material form into the one that is stored on the device.
Only `MixMaterialSpec` needs it today: its sub-materials must be pushed first
so the mix can reference them by `SetKey`.
"""
resolve_material(scene, mat::Material) = mat
function resolve_material(scene, spec::MixMaterialSpec)
    key1 = push!(scene.materials, resolve_material(scene, spec.material1))
    key2 = push!(scene.materials, resolve_material(scene, spec.material2))
    return MixMaterial(spec.amount, key1, key2)
end

# MixMaterial is not directly emissive (emission comes from chosen sub-material)
is_emissive(::MixMaterial) = false

# ============================================================================
# Hash function for deterministic material selection
# ============================================================================

"""
    mix_hash_float(p::Point3f, wo::Vec3f, idx1::SetKey, idx2::SetKey) -> Float32

Generate a deterministic pseudo-random float in [0, 1) for material selection.
Uses a simple but effective hash function based on pbrt-v4's HashFloat.

The hash is deterministic: same position, direction, and materials always
produce the same result, ensuring consistent rendering across samples.
"""
@propagate_inbounds function mix_hash_float(
    p::Point3f, wo::Vec3f,
    idx1::SetKey, idx2::SetKey
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
    h = xor(h, UInt64(idx1.type_idx) << 24)
    h = xor(h, UInt64(idx1.vec_idx))
    h = h * UInt64(0xcc9e2d51)
    h = xor(h, UInt64(idx2.type_idx) << 28)
    h = xor(h, UInt64(idx2.vec_idx) << 4)
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
    choose_material(mix::MixMaterial, textures, p::Point3f, wo::Vec3f, tfc) -> SetKey

Choose which sub-material to use at the given hit point.
Returns the SetKey of the chosen material.

Following pbrt-v4's ChooseMaterial:
1. Evaluate the amount texture at the hit (`tfc`: uv, face and barycentrics)
2. If amount ≤ 0, return material1
3. If amount ≥ 1, return material2
4. Otherwise, use deterministic hash to stochastically select

This function is called at intersection time, before material evaluation.
"""
@propagate_inbounds function choose_material(
    mix::MixMaterial, ctx::StaticMultiTypeSet,
    p::Point3f, wo::Vec3f, tfc::TextureFilterContext
)::SetKey
    amt = eval_handle(ctx, mix.amount, tfc)

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
    is_mix_material_dispatch(materials, idx::SetKey) -> Bool

Type-stable dispatch to check if a material is MixMaterial.
"""
@propagate_inbounds function is_mix_material_dispatch(
    materials::StaticMultiTypeSet, idx::SetKey
)::Bool
    # A material that is not in the set is not a mix. Guarding here rather than
    # asking `with_index`, which has no element to hand `is_mix_material` and
    # would either fall into its else-branch — answering about an ARBITRARY
    # material — or, when the set is empty, expand to a bare `error(...)` that a
    # GPU kernel cannot raise and simply dies on. See
    # `get_surface_alpha_dispatch` for the same guard and what it cost.
    Raycore.is_valid(idx) || return false
    return with_index(is_mix_material, materials, idx)
end

# Helper for resolve_mix_material - called with concrete material type
@propagate_inbounds function choose_material_impl(mat, ctx, p, wo, tfc, idx::SetKey)
    return is_mix_material(mat) ? choose_material(mat, ctx, p, wo, tfc) : idx
end

"""
    resolve_mix_material(materials::StaticMultiTypeSet, idx::SetKey, p, wo, tfc) -> SetKey

Resolve any MixMaterial chain to get the final material index.
Handles nested MixMaterials by iterating until a non-mix material is found.
`materials` is used both for material lookup and texture evaluation.

This should be called at intersection time before creating material work items.
"""
@propagate_inbounds function resolve_mix_material(
    materials::StaticMultiTypeSet,
    idx::SetKey,
    p::Point3f, wo::Vec3f, tfc::TextureFilterContext
)::SetKey
    # Iterate to handle nested MixMaterials (up to 8 levels to prevent infinite loops)
    current_idx = idx
    for _ in 1:8
        if !is_mix_material_dispatch(materials, current_idx)
            return current_idx
        end
        current_idx = with_index(choose_material_impl, materials, current_idx, materials, p, wo, tfc, current_idx)
    end
    return current_idx
end

# NOTE: MixMaterial does NOT implement spectral evaluation functions.
# Following pbrt-v4, MixMaterial is always resolved at intersection time
# to a concrete material before any BSDF evaluation occurs.
# See pbrt-v4 materials.h line 339-344: GetBxDF() is LOG_FATAL if called.

