# The coefficient table's z interval, found as pbrt-v4's `FindInterval` finds it
# (`find_interval`). It was a scan over all 63 nodes for every colour lifted to a
# spectrum; the binary search has to name the interval the scan named, for every z:
# at the nodes, next to them, between them and at both ends.
using Test, Hikari
using GeometryBasics: Point2f

@testset "find_interval names the interval a scan would" begin
    table = Hikari.get_srgb_table()
    nodes, n = table.scale, table.res
    @test issorted(nodes)
    scan(z) = something(findlast(i -> nodes[i] < z, 1:(Int(n) - 1)), 1)
    zs = Float32[0, 1, nodes..., prevfloat.(nodes)..., nextfloat.(nodes)..., range(0f0, 1f0; length = 10_001)...]
    @test all(z -> Hikari.find_interval(i -> nodes[i] < z, n - Int32(1)) == scan(z), zs)
    # The CDF searches share the core: the last bin a CDF value of `u` starts.
    cdf = Float32[0, 0.1, 0.1, 0.5, 1]
    lastbin(u) = findlast(i -> cdf[i] <= u, 1:length(cdf))
    @test all(u -> Hikari.find_interval(i -> cdf[i] <= u, Int32(length(cdf))) == lastbin(u),
              Float32[0, 0.05, 0.1, 0.3, 0.5, 0.99, 1])
end

# A closure a search takes must not capture a variable assigned twice: Julia boxes
# it, and a boxed capture is a dynamic call in device code. The environment map's
# column search captured its row offset, assigned by the row search and again by a
# clamp; Mantle refused the shade kernel ("a call to `Base.<` ... could not resolve").
@testset "the samplers' searches capture nothing boxed" begin
    d = Hikari.Distribution2D(Float32[1 2 3; 4 5 6; 7 8 9; 1 1 1])
    typed = only(code_typed(Hikari.sample_continuous, (typeof(d), Point2f, Nothing)))
    @test !occursin("Core.Box", sprint(show, typed))
    table = Hikari.get_srgb_table()
    typed = only(code_typed(Hikari.rgb_to_spectrum, (typeof(table), Float32, Float32, Float32)))
    @test !occursin("Core.Box", sprint(show, typed))
end
