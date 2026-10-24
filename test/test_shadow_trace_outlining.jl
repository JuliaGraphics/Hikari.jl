using Test
using Hikari, Mantle

@testset "Shadow trace is outlined only around an inline ray query" begin
    descriptor_bound = Mantle.AdaptedAccel(nothing, Int32[], Int32[], nothing, nothing, nothing, nothing)
    argument_buffer = Mantle.AdaptedAccel(nothing, Int32[], Int32[], nothing, UInt8[], nothing, nothing)

    @test !Hikari.traverses_by_linked_call(descriptor_bound)
    @test Hikari.traverses_by_linked_call(argument_buffer)
    @test !Hikari.traverses_by_linked_call(nothing)
end
