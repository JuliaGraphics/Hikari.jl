using Test, Hikari

# The reference scenes share their ground and sphere through `Include`, resolved
# against the including file's directory as in pbrt-v4. A missing file was only
# a warning, so a scene silently lost that geometry; pbrt-v4 stops instead.
@testset "pbrt Include" begin
    dir = mktempdir()
    triangle = """Shape "trianglemesh" "point3 P" [0 0 0  1 0 0  0 1 0] "integer indices" [0 1 2]\n"""
    mkpath(joinpath(dir, "geometry"))
    write(joinpath(dir, "geometry", "triangle.pbrt"), triangle)
    write(joinpath(dir, "inline.pbrt"), "WorldBegin\n" * triangle)
    write(joinpath(dir, "included.pbrt"), "WorldBegin\nInclude \"geometry/triangle.pbrt\"\n")
    write(joinpath(dir, "missing.pbrt"), "WorldBegin\nInclude \"geometry/none.pbrt\"\n")

    inline = Hikari.parse_pbrt(joinpath(dir, "inline.pbrt"))
    included = Hikari.parse_pbrt(joinpath(dir, "included.pbrt"))
    @test length(included.shapes) == 1
    @test included.shapes[1].entity.params["P"].values == inline.shapes[1].entity.params["P"].values
    @test included.shapes[1].entity.params["indices"].values == inline.shapes[1].entity.params["indices"].values

    @test_throws ErrorException Hikari.parse_pbrt(joinpath(dir, "missing.pbrt"))
end
