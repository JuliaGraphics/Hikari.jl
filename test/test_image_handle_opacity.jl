# An image handle says whether its image is opaque. RayMakie stores a plot's images
# before Hikari builds the shader, and an image handle always counted as a cutout,
# so every image-textured Diffuse carried an alpha program it never needed: on an
# RTX 4000 Ada that cost killeroo-gold's floor 21% of a frame.
using Test, Hikari, Raycore, Colors
using KernelAbstractions: CPU

opaque_image() = [RGBA{Float32}(0.5, 0.25, 0.75, 1) for _ in 1:4, _ in 1:4]
function holey_image()
    img = opaque_image()
    img[2, 3] = RGBA{Float32}(0.5, 0.25, 0.75, 0.5)
    return img
end
alpha_program(material) = Hikari.surface_material(material).alpha

@testset "an image handle knows whether its image is opaque" begin
    store = Raycore.MultiTypeSet(CPU())
    solid = Hikari.device_param(store, opaque_image())
    holey = Hikari.device_param(store, holey_image())
    @test solid.kind == Hikari.TexKind.IMAGE && holey.kind == Hikari.TexKind.IMAGE
    @test !Hikari.cutout(solid)
    @test Hikari.cutout(holey)

    # A material holding the stored image reads alpha only where the image has some.
    @test alpha_program(Hikari.Diffuse(color = solid)) === nothing
    @test alpha_program(Hikari.Diffuse(color = holey)) !== nothing
    @test alpha_program(Hikari.CoatedDiffuse(reflectance = solid)) === nothing
    @test alpha_program(Hikari.CoatedDiffuse(reflectance = holey)) !== nothing

    # An update in place keeps the slot and follows the new pixels.
    now_holey = Raycore.update_item(store, solid, Hikari.to_texture(holey_image()))
    @test (now_holey.slot, now_holey.idx) == (solid.slot, solid.idx)
    @test Hikari.cutout(now_holey)
    now_solid = Raycore.update_item(store, now_holey, Hikari.to_texture(opaque_image()))
    @test (now_solid.slot, now_solid.idx) == (solid.slot, solid.idx)
    @test !Hikari.cutout(now_solid)

    # A handle made from a bare reference cannot read its pixels.
    ref = Raycore.store_texture(store, Hikari.to_texture(opaque_image()).data)
    @test Hikari.cutout(Hikari.TexHandle(ref))
end
