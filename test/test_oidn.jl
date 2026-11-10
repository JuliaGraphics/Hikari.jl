module OIDNTests
using Test, Hikari, Mantle
using GeometryBasics: Point2f

# A film's OIDN device and filter hold about 250 MB of the library's own memory at
# 1200x360. They were released only by `free!(film)`, so every film dropped after
# a denoise kept them: the docs build denoises about twenty films, grew by 5 GB
# and peaked at the CI runner's 16 GB. The plan now releases them when it is collected.
#
# Hikari ships OIDN's official Linux x86_64 build only.
if Sys.islinux() && Sys.ARCH === :x86_64
    @testset "a dropped OIDN plan releases its device and filter" begin
        backend = Mantle.defaultbackend()
        film = Hikari.Film(backend, Hikari.Film(Point2f(16, 8)))
        Hikari.denoise!(film; config = Hikari.OIDN(guides = false))
        plan = film.denoise_plan[]
        @test plan isa Hikari.OIDNPlan
        @test plan.device != C_NULL && plan.filter != C_NULL
        # What the GC does to a plan nothing references any more.
        finalize(plan)
        @test plan.device == C_NULL && plan.filter == C_NULL
        # `free!` after the finalizer is a no-op, not a double release.
        Hikari.free!(film)
        @test film.denoise_plan[] === nothing
    end
end
end
