using Test
using Hikari, Mantle

module PrecompileDeviceProbe
using Mantle
struct API <: Mantle.Backend end
struct Device <: Mantle.Device end
const constructions = Ref(0)
function Mantle.makedevice(::API, select, debug::Mantle.DebugConfig)
    constructions[] += 1
    return Device()
end
end

# The workload infers concrete device and scene types. It must not initialize
# a device: precompiling a renderer cannot depend on working GPU hardware.
@testset "precompile statements all resolve" begin
    @test PrecompileDeviceProbe.Device in Hikari._precompile_device_types()
    @test PrecompileDeviceProbe.constructions[] == 0
    ok, total = Hikari._precompile_statements()
    @test total > 0
    @test ok == total
    @test PrecompileDeviceProbe.constructions[] == 0
end
