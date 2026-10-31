module RuntimeBSDFRenderTests
using Test, Hikari, Mantle, GeometryBasics
using Hikari.TextureNodes
using GeometryBasics: Point2f, Point3f, Vec3f, Sphere, Tesselation, normal_mesh

function material(order; computed = false, bump_normal = false)
    color = computed ? noise(xyz(); scale = 3).color : (0.4, 0.2, 0.1)
    normal = bump_normal ? bump(noise(xyz(); scale = 6).fac; strength = 0.1) : nothing
    nodes = (Hikari.Diffuse(; color, normal, roughness = 0.4),
             Hikari.Sheen(; normal, color = (0.25, 0.3, 0.4), roughness = 0.35),
             Hikari.Glossy(; normal, color = (0.3, 0.3, 0.3), roughness = 0.25))
    return foldl((a, b) -> Hikari.Mix(a, b; factor = 0.35), (nodes[i] for i in order))
end

function fixture(desc; hardware = true)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend, hw_accel = hardware)
    handle = push!(scene, normal_mesh(Tesselation(Sphere(Point3f(0), 0.7f0), 24)), desc)
    push!(scene, Hikari.PointLight(Point3f(-2, -2, 3), Hikari.RGBSpectrum(40f0)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(48, 48)))
    camera = Hikari.PerspectiveCamera(Point3f(0, -3, 1), Point3f(0), film; fov = 40f0)
    vp = Hikari.VolPath(samples = 4, max_depth = 4)
    return (; scene, handle, film, camera, vp, dev = Hikari.mantle_device(backend))
end
draw(f) = (f.vp(f.scene, f.film, f.camera); Array(f.film.framebuffer))
release(f) = (close(f.vp); Hikari.free!(f.film); nothing)
signatures(f) = unique([Mantle.devicetype(f.dev, d.args)
    for pl in Hikari.allplans(f.vp.state.plans) for pp in pl.passes
    if pp.pass.name == "shade" for d in pp.pass.dispatches])

# Select the retained typed implementation only for the reference's plan. The production
# selector is restored even when rendering throws; no configuration is left in the package.
function reference_image(f)
    @eval Hikari.bsdf_kernel_nodes(::Type{M}, ::Mantle.Backend) where {M <: Tuple} = M
    method = which(Hikari.bsdf_kernel_nodes, Tuple{Type{Tuple{}}, Mantle.VulkanAPI})
    try
        return Base.invokelatest(draw, f)
    finally
        Base.delete_method(method)
    end
end

@testset "runtime BSDF render parity" begin
    for (order, computed, bump_normal, hardware) in (
            ((1, 2, 3), false, false, true),
            ((1, 2, 3), true, false, true),
            ((1, 2, 3), false, true, true),
            ((1, 2, 3, 1, 2, 3, 1, 2), false, false, true),
            ((1, 2, 3), false, false, false),
        )
        desc = material(order; computed, bump_normal)
        f = fixture(desc; hardware)
        try
            expected = reference_image(f)
            # A new VolPath records the production kernel class over the same scene.
            close(f.vp)
            f = merge(f, (; vp = Hikari.VolPath(samples = 4, max_depth = 4)))
            actual = Base.invokelatest(draw, f)
            @test any(p -> p.r > 0, actual)
            @test actual == expected
            @test any(T -> occursin("RuntimeBSDFNodes", string(T)), signatures(f))
            if length(order) == 3 && !computed && !bump_normal && hardware
                before = signatures(f)
                for permutation in ((3, 1, 2), (2, 3, 1))
                    Hikari.update_material!(f.scene, f.handle, material(permutation))
                    @test all(p -> isfinite(p.r) && isfinite(p.g) && isfinite(p.b), Base.invokelatest(draw, f))
                    @test signatures(f) == before
                end
            end
        finally
            release(f)
        end
    end
end
end
