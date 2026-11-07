# Regression: `update_material!` on a `MediumInterface{NullMaterial,...}` used
# to crash with `BoundsError: attempt to access 1-element Vector{DataType} at
# index [0]` because:
#
#   1. `push!(scene, MediumInterface(NullMaterial(); inside=medium))` stores
#      the `NullMaterial` via `push!(::MultiTypeSet, ::NullMaterial) = SetKey()`
#      → `mi.material == SetKey(0, 0)` (the invalid sentinel).
#   2. `update_material!(scene, idx, new_material::Material)` fell through the
#      `::Material` overload and called
#      `Raycore.update!(scene.materials, mi.material, new_material)`.
#   3. `Raycore.update!` indexed `dhv.data_order[0]` → BoundsError.
#
# This hit RayMakie's mesh-swap path on every frame of a volume render whose
# bounding mesh had `MediumInterface(NullMaterial(); inside=medium)`, killing
# the HQ glow video at frame 2. The two-layer fix:
#   - Raycore `update!(::MultiTypeSet, ::SetKey, _)` short-circuits on
#     `is_invalid(key)` (unpushed slot = nothing to refresh).
#   - Hikari `update_material!(::Scene, ::UInt32, ::MediumInterface)` is its
#     own overload that recurses into the three component slots (inner
#     material + inner/outer media), each guarded by `Raycore.is_valid` so
#     unpushed sides (NullMaterial, `inside=nothing`, `outside=nothing`)
#     are silent no-ops.
using Test
using Hikari
using Raycore
using Lava, Mantle
using GeometryBasics
using GPUArraysCore: @allowscalar

const _BACKEND = Mantle.defaultbackend()

function _mini_scene_with_null_medium_cube()
    scene = Hikari.Scene(; backend=_BACKEND)
    medium = Hikari.RGBGridMedium(
        σ_a_grid=fill(Hikari.RGBSpectrum(0f0),  4, 4, 4),
        σ_s_grid=fill(Hikari.RGBSpectrum(0.1f0), 4, 4, 4),
        sigma_scale=1f0, g=0f0,
        bounds=Raycore.Bounds3(Point3f(0, 0, 0), Point3f(1, 1, 1)))
    cube = GeometryBasics.normal_mesh(Rect3f(Vec3f(0), Vec3f(1)))
    mi   = Hikari.MediumInterface(Hikari.NullMaterial(); inside=medium)
    handle = push!(scene, cube, mi)
    return scene, handle, medium
end

@testset "update_material! on MediumInterface{NullMaterial} does not BoundsError" begin
    scene, handle, old_medium = _mini_scene_with_null_medium_cube()

    # Swap to a fresh medium via a NEW MediumInterface with the same
    # NullMaterial + no outside — exact shape of the RayMakie mesh-swap call.
    new_medium = Hikari.RGBGridMedium(
        σ_a_grid=fill(Hikari.RGBSpectrum(0f0),  4, 4, 4),
        σ_s_grid=fill(Hikari.RGBSpectrum(0.5f0), 4, 4, 4),  # different σ_s
        sigma_scale=1f0, g=0f0,
        bounds=Raycore.Bounds3(Point3f(0, 0, 0), Point3f(1, 1, 1)))
    new_mi = Hikari.MediumInterface(Hikari.NullMaterial(); inside=new_medium)

    # Pre-fix: this throws BoundsError. Post-fix: silent update, inside-medium
    # slot sees the new σ_s values.
    @test (Hikari.update_material!(scene, handle.interface, new_mi); true)
end

@testset "update_material! on MediumInterface still refreshes the inside medium slot" begin
    # Positive check: when the inside medium IS validly pushed (which it is
    # for any non-nothing Medium), its slot contents actually get updated so
    # that rendering sees the new σ_s. Reads the stored σ_s_grid back
    # through the Raycore deref path so TextureRef gets unwrapped to its
    # underlying array.
    scene, handle, old_medium = _mini_scene_with_null_medium_cube()
    mi = @allowscalar scene.media_interfaces[handle.interface]
    @test Raycore.is_valid(mi.inside)
    n_media_before = length(scene.media)

    new_medium = Hikari.RGBGridMedium(
        σ_a_grid=fill(Hikari.RGBSpectrum(0f0),  4, 4, 4),
        σ_s_grid=fill(Hikari.RGBSpectrum(0.7f0), 4, 4, 4),
        sigma_scale=1f0, g=0f0,
        bounds=Raycore.Bounds3(Point3f(0, 0, 0), Point3f(1, 1, 1)))
    new_mi = Hikari.MediumInterface(Hikari.NullMaterial(); inside=new_medium)
    Hikari.update_material!(scene, handle.interface, new_mi)

    # Slot count unchanged — refresh in place, no accidental re-push.
    @test length(scene.media) == n_media_before

    # Inner medium slot should reflect the new σ_s value. Raycore stores
    # `σ_a_grid` and `σ_s_grid` as `TextureRef`s pointing into the
    # `scene.media.texture_gpu_arrays` flat vector (push order). For a single
    # RGBGridMedium, `maybe_convert_field` walks struct fields in
    # declaration order (σ_a_grid, σ_s_grid, Le_grid, majorant_grid), so the
    # σ_s LavaArray is entry 2.  Reading back via Array() gives us the
    # authoritative post-update contents.
    σ_s_lava = scene.media.texture_gpu_arrays[2]
    @test (@allowscalar Array(σ_s_lava)[1, 1, 1].c[1]) ≈ 0.7f0
end

@testset "update_material! on MediumInterface is in-place (no slot duplication)" begin
    # Guard against an alternative regression where update_material! quietly
    # re-pushes the medium (which would leak a slot per frame in a render
    # loop). The inside-medium TextureRef identity must stay stable — same
    # type slot, same element index — and `length(scene.media)` must not
    # grow across multiple updates.
    scene, handle, old_medium = _mini_scene_with_null_medium_cube()
    mi = @allowscalar scene.media_interfaces[handle.interface]
    CT = scene.media.data_order[mi.inside.type_idx]

    tref_before = scene.media.data_vectors[CT][mi.inside.vec_idx].σ_s_grid
    n_media_before    = length(scene.media)
    n_texarrs_before  = length(scene.media.texture_gpu_arrays)

    for σ_s_val in (0.2f0, 0.3f0, 0.4f0)
        new_medium = Hikari.RGBGridMedium(
            σ_a_grid=fill(Hikari.RGBSpectrum(0f0),  4, 4, 4),
            σ_s_grid=fill(Hikari.RGBSpectrum(σ_s_val), 4, 4, 4),
            sigma_scale=1f0, g=0f0,
            bounds=Raycore.Bounds3(Point3f(0, 0, 0), Point3f(1, 1, 1)))
        Hikari.update_material!(scene, handle.interface,
                                Hikari.MediumInterface(Hikari.NullMaterial(); inside=new_medium))
    end

    tref_after = scene.media.data_vectors[CT][mi.inside.vec_idx].σ_s_grid
    @test typeof(tref_after) === typeof(tref_before)  # same TIdx (type slot)
    @test tref_after.idx == tref_before.idx           # same element index
    @test length(scene.media)                       == n_media_before
    @test length(scene.media.texture_gpu_arrays)    == n_texarrs_before
end
