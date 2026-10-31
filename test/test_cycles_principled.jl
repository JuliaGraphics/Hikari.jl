# The Principled BSDF against Blender 5.2.1's Cycles, input by input, as #47 checked it: the mean
# colour of each sphere of `cycles/principled_cases.jl` against Cycles' (OptiX, 4096 samples, filter
# glossy off; how they were made is in `cycles/export_scene.jl`), within a tolerance per case.
#
# Most cases agree within 0.3 % at 512 samples. The tolerance is that agreement at 64 samples and at
# 512, whichever is further off, plus 1 %, at least 1.5 %. Further off: coloured transmission and
# subsurface, where light that passes through the material several times takes its colour each time
# and the product of uplifted spectra is not the uplift of the RGB product (spectral vs RGB, up to
# 8 % in one channel), and the colour of a rough metal's multiple scattering (3 %).
#
# What this comparison found: every dielectric Principled rendered with a black diffuse base (-95 %)
# when the shading kernels' view of the material set had no `lookup_tables` method; and, with
# regularization on, a rough glass lost 3 % of the light (`test_white_furnace.jl` pins that).
# Regularization is off here, as Cycles' filter glossy is off in the references.
using Test, Hikari, Mantle, GeometryBasics
using GeometryBasics: Point2f
include(joinpath(@__DIR__, "cycles", "principled_cases.jl"))

const CYCLES_REFERENCE = Dict(
    "default" =>            ((0.29463, 0.29076, 0.29076), 0.015),
    "base_red" =>           ((0.36839, 0.08958, 0.08958), 0.015),
    "rough0" =>             ((0.36729, 0.36382, 0.36382), 0.015),
    "rough1" =>             ((0.28995, 0.28971, 0.28971), 0.015),
    "diffuse_rough" =>      ((0.17979, 0.1797, 0.1797), 0.015),
    "ior_low" =>            ((0.22242, 0.22234, 0.22234), 0.015),
    "ior_high" =>           ((0.23719, 0.23708, 0.23707), 0.015),
    "alpha" =>              ((0.16685, 0.16669, 0.16669), 0.015),
    "metal" =>              ((0.36536, 0.24063, 0.08042), 0.015),
    "metal_rough" =>        ((0.40761, 0.25379, 0.09692), 0.04),
    "metal_tint" =>         ((0.44553, 0.30398, 0.11969), 0.015),
    "aniso" =>              ((0.30113, 0.29865, 0.29477), 0.015),
    "aniso_rot" =>          ((0.23738, 0.23712, 0.2368), 0.015),
    "spec0" =>              ((0.2222, 0.22209, 0.22197), 0.015),
    "spec1" =>              ((0.22898, 0.22882, 0.22869), 0.015),
    "spec_tint" =>          ((0.20713, 0.19989, 0.19973), 0.015),
    "glass" =>              ((0.13456, 0.13467, 0.13525), 0.015),
    "glass_rough" =>        ((0.15627, 0.16166, 0.17646), 0.015),
    "glass_color" =>        ((0.13099, 0.06919, 0.07235), 0.055),
    "glass_gray" =>         ((0.0892, 0.08538, 0.08334), 0.015),
    "half_transmission" =>  ((0.14287, 0.14336, 0.14467), 0.02),
    "coat" =>               ((0.05003, 0.10068, 0.22904), 0.015),
    "coat_rough" =>         ((0.04834, 0.09887, 0.22698), 0.015),
    "coat_tint" =>          ((0.18628, 0.10543, 0.04793), 0.015),
    "coat_ior" =>           ((0.09626, 0.15806, 0.29516), 0.025),
    "coat_metal" =>         ((0.5161, 0.33698, 0.14287), 0.025),
    "sheen" =>              ((0.12541, 0.27918, 0.11698), 0.025),
    "sheen_rough" =>        ((0.08039, 0.2209, 0.0789), 0.015),
    "sheen_tint" =>         ((0.07335, 0.1622, 0.05851), 0.025),
    "emission" =>           ((1.23198, 0.72835, 0.42494), 0.015),
    "thin_film" =>          ((0.15663, 0.1476, 0.13983), 0.02),
    "thin_film_metal" =>    ((0.23347, 0.22684, 0.1888), 0.02),
    "sss" =>                ((0.31039, 0.26051, 0.21398), 0.085),
    "sss_gray" =>           ((0.36824, 0.35854, 0.352), 0.02),
    "sss_aniso" =>          ((0.25976, 0.25859, 0.25909), 0.015),
    "sss_ior" =>            ((0.29103, 0.2905, 0.29704), 0.015),
    "sss_scale" =>          ((0.20591, 0.20787, 0.16565), 0.095),
    "sss_half" =>           ((0.26108, 0.1988, 0.1682), 0.045),
    "glass_spec" =>         ((0.15865, 0.15769, 0.15841), 0.015),
    "rough_spec" =>         ((0.14235, 0.14185, 0.21212), 0.015),
)

function render_cycles_batch(cases; samples)
    backend = Mantle.defaultbackend()
    scene = Hikari.Scene(; backend)
    for ((name, inputs), c) in zip(cases, CYCLES_CENTERS)
        push!(scene, cycles_sphere(c), Hikari.Principled(; inputs...))
    end
    push!(scene, cycles_ground(), Hikari.Diffuse(color = (0.5, 0.5, 0.5)))
    push!(scene, cycles_light(), Hikari.Emission(strength = CYCLES_SCENE.light.strength, two_sided = true))
    push!(scene, Hikari.AmbientLight(Hikari.RGBSpectrum(CYCLES_SCENE.world)))
    Hikari.sync!(scene)
    film = Hikari.Film(backend, Hikari.Film(Point2f(CYCLES_SCENE.res, CYCLES_SCENE.res)))
    cam = CYCLES_SCENE.camera
    camera = Hikari.PerspectiveCamera(cam.position, cam.lookat, film; up = cam.up, fov = cam.fov)
    vp = Hikari.VolPath(; samples, max_depth = CYCLES_SCENE.max_depth, regularize = false)
    vp(scene, film, camera)
    img = Array(film.framebuffer)
    close(vp)
    return img
end

@testset "Principled against Cycles" begin
    pixels = [cycles_sphere_pixels(k) for k in eachindex(CYCLES_CENTERS)]
    for cases in cycles_batches()
        img = render_cycles_batch(cases; samples = 64)
        for (k, (name, _)) in enumerate(cases)
            reference, tolerance = CYCLES_REFERENCE[name]
            mean = cycles_mean(img, pixels[k])
            @testset "$name" begin
                for ch in 1:3
                    @test mean[ch] ≈ reference[ch] rtol = tolerance
                end
            end
        end
    end
end
