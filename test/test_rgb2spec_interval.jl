# The coefficient table's z interval, found as pbrt-v4's `FindInterval` finds it
# (`find_interval`). It was a scan over all 63 nodes for every colour lifted to a
# spectrum; the binary search has to name the interval the scan named, for every z:
# at the nodes, next to them, between them and at both ends.
using Test, Hikari

@testset "find_interval names the interval a scan would" begin
    table = Hikari.get_srgb_table()
    nodes, n = table.scale, table.res
    @test issorted(nodes)
    scan(z) = something(findlast(i -> nodes[i] < z, 1:(Int(n) - 1)), 1)
    zs = Float32[0, 1, nodes..., prevfloat.(nodes)..., nextfloat.(nodes)..., range(0f0, 1f0; length = 10_001)...]
    @test all(z -> Hikari.find_interval(nodes, n, z) == scan(z), zs)
end
