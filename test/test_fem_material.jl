"""
`FEMMaterial`'s public surface: the constructor a user actually types, and what
happens when the backend cannot trace the geometry it carries.

Neither is about rendering, which is why this file is cheap and runs anywhere.
Both were found by trying to build one on a Mac, and both failed in a way that
named nothing useful.
"""

using Test, Hikari, Mantle

@testset "FEMMaterial" begin
    # Two trivial cells: the identity map in x and y, a constant field. Enough
    # to construct — the geometry is not traced here.
    cx = zeros(Float64, Hikari.FEM_NMONO, 2)
    cy = zeros(Float64, Hikari.FEM_NMONO, 2)
    cf = zeros(Float64, Hikari.FEM_NMONO, 2)
    cx[2, :] .= 1.0
    cy[3, :] .= 1.0
    cf[1, :] .= 0.5

    @testset "the documented constructor works on its own defaults" begin
        # `FEMMaterial(cx, cy, cf)` is the first form in the docstring and it
        # could not run: `colormap` defaulted to `:viridis`, and
        # `FEMFieldTexture` takes "any Makie-style colour ramp", which is a
        # VECTOR. Hikari has `Colors` and no colormap package, so nothing could
        # turn the symbol into colours — every caller that worked passed a ramp,
        # so the default was never exercised.
        m = Hikari.FEMMaterial(cx, cy, cf; warp = 0.6)
        @test m isa Hikari.FEMMaterial
        @test length(Hikari.FEM_DEFAULT_RAMP) > 1

        # …and an explicit ramp still works, of any length: it is resampled.
        m2 = Hikari.FEMMaterial(cx, cy, cf; colormap = Hikari.FEM_DEFAULT_RAMP[1:8])
        @test m2 isa Hikari.FEMMaterial
    end

    @testset "a colormap NAME declines and says what to pass" begin
        # Not a `MethodError` listing two unrelated candidates. The message has
        # to name the alternative, because "pass a vector" is not guessable from
        # a signature mismatch.
        err = try
            Hikari.FEMMaterial(cx, cy, cf; colormap = :viridis); nothing
        catch e; e end
        @test err isa ErrorException
        @test occursin("NAME", err.msg)
        @test occursin("vector of colours", err.msg)
    end

    @testset "an update stores what the push stored" begin
        # `push!` registers the SURFACE with the field merged in, and an update
        # handed the `FEMMaterial` wrapper tried to `convert` it into that
        # surface's slot — `Cannot convert FEMMaterial to ThinDielectric` for
        # glass, and the same for every surface. RayMakie updates a plot's
        # material this way: the isubd demo's tolerance slider did, and the error
        # killed the render loop on the next switch to RASTER.
        backend = Mantle.defaultbackend()
        if Mantle.supports_procedural_traversal(backend)
            # One per slot type the demo uses, because each is stored apart.
            for surface in (Hikari.CoatedDiffuse(roughness = 0.1), Hikari.ThinDielectric(),
                            Hikari.Conductor())
                scene = Hikari.Scene(; backend = backend, hw_accel = true)
                h = push!(scene, Hikari.FEMMaterial(surface, cx, cy, cf; warp = 0.6,
                                                    tolerance = 2f-2))
                # Same types, new tolerance: exactly what the slider sends.
                newer = Hikari.FEMMaterial(surface, cx, cy, cf; warp = 0.6, tolerance = 5f-3)
                @test (Hikari.update_material!(scene, h.interface, newer); true)
            end
        end
    end

    @testset "pushing it asks whether the backend can trace a box" begin
        # ONE assertion, same meaning on every backend: a scene accepts
        # procedural geometry exactly when the backend can traverse it.
        #
        # The two halves come apart, and that is the whole point of the
        # predicate. Building an AABB BLAS and registering the instance succeeds
        # on Metal; tracing it does not exist there. Without the question, every
        # step of `push!` succeeds and the failure lands later as a `MethodError`
        # on `candidate_object_ray` inside a shader compile, naming neither the
        # geometry nor the backend nor the reason.
        backend = Mantle.defaultbackend()
        # `hw_accel = true`: procedural geometry is a BLAS of boxes, and only a
        # hardware acceleration structure takes one. A software TLAS holds
        # triangles and has no `push!` for a BLAS at all.
        scene = Hikari.Scene(; backend = backend, hw_accel = true)
        mat = Hikari.FEMMaterial(cx, cy, cf; warp = 0.6)
        can = Mantle.supports_procedural_traversal(backend)

        if can
            @test push!(scene, mat) isa Hikari.SceneHandle

            # …and a SOFTWARE structure declines on every backend, because it
            # holds triangles and takes no BLAS. This used to be a bare
            # `MethodError` on `push!` naming two types and no reason — the
            # question is asked of the ACCEL for exactly this case, not of the
            # backend, which cannot tell the two mistakes apart.
            soft = Hikari.Scene(; backend = backend)
            @test !Mantle.supports_procedural_traversal(soft.accel)
            softerr = try; push!(soft, mat); nothing catch e; e end
            @test softerr isa ErrorException
            @test occursin("hw_accel = true", softerr.msg)
        else
            err = try; push!(scene, mat); nothing catch e; e end
            @test err isa ErrorException
            # It names the predicate, so the reader can check it themselves.
            @test occursin("supports_procedural_traversal", err.msg)
            # …and points at the path that DOES work for these elements.
            @test occursin("raster", err.msg)
        end
    end
end
