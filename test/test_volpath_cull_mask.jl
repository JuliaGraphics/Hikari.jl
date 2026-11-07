using Test, Hikari

@testset "VolPath cull_mask field default + override" begin
    # Default field value -- zero-arg form uses all defaults
    vp = Hikari.VolPath()
    @test vp.cull_mask == UInt32(0xFF)

    # Override via kwarg
    vp_masked = Hikari.VolPath(cull_mask=UInt32(0x04))
    @test vp_masked.cull_mask == UInt32(0x04)
end
