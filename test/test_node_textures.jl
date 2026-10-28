using Test
using Hikari
using Hikari.TextureNodes
using Hikari.TextureNodes: magic   # FileIO, which the suite loads, exports a `magic` too
using Hikari: TextureFilterContext, TexKind, RGBSpectrum, device_param, eval_handle, eval_handle_spectrum
using GeometryBasics
using GeometryBasics: Point2f, Point3f, Vec3f
using KernelAbstractions: CPU
import KernelAbstractions as KA
using LinearAlgebra
using Raycore
using Mantle
using Statistics: mean

"""The store's texture table on the host: adapting to `CPU()` would give POCL device arrays."""
host_static(store) =
    Raycore.StaticMultiTypeSet((), Tuple(filter(a -> typeof(a) === T, store.texture_gpu_arrays) for T in store.texture_order))

function evaluate(graph; uv = Point2f(0), p = Point3f(0), n = Vec3f(0, 0, 1))
    store = Raycore.MultiTypeSet(CPU())
    handle = device_param(store, graph)
    tfc = TextureFilterContext(uv, p, n)
    rgb = eval_handle_spectrum(host_static(store), handle, tfc)
    return Vec3f(rgb.c[1], rgb.c[2], rgb.c[3])
end

at(graph, p) = evaluate(graph; p = Point3f(p))
scalar_at(graph, p) = at(graph, p)[1]

const CYCLES_GRAPHS = Dict(
    "noise_fbm3" => p -> noise(p),
    "noise_fbm3_color" => p -> noise(p).color,
    "noise_fbm3_frac" => p -> noise(p; detail = 3.6, roughness = 0.62, lacunarity = 2.3),
    "noise_distort4" => p -> noise(p; dimensions = 4, w = 0.7, distortion = 0.5).color,
    "noise_ridged3" => p -> noise(p; type = :ridged_multifractal, detail = 4.5, offset = 1, gain = 2),
    "noise_hetero2" => p -> noise(p; dimensions = 2, type = :hetero_terrain, detail = 3, offset = 0.4),
    "noise_multi1" => p -> noise(p; dimensions = 1, type = :multifractal, w = 1.3, detail = 2.5),
    "noise_hybrid3_raw" => p -> noise(p; type = :hybrid_multifractal, normalize = false, detail = 3, gain = 1.5, offset = 0.2),
    "voronoi_f1" => p -> voronoi(p),
    "voronoi_f2_manhattan" => p -> voronoi(p; feature = :f2, metric = :manhattan),
    "voronoi_smooth" => p -> voronoi(p; feature = :smooth_f1, smoothness = 0.6).color,
    "voronoi_edge" => p -> voronoi(p; feature = :distance_to_edge),
    "voronoi_fractal" => p -> voronoi(p; normalize = true, detail = 2.5, roughness = 0.6, randomness = 0.8),
    "voronoi_minkowski2" => p -> voronoi(p; dimensions = 2, metric = :minkowski, exponent = 1.5),
    "voronoi_4d" => p -> voronoi(p; dimensions = 4, w = 0.3).color,
    "voronoi_1d" => p -> voronoi(p; dimensions = 1, w = 0.9, scale = 3),
    "gradient_radial" => p -> gradient(p; type = :radial).fac,
    "checker" => p -> checker(p; scale = 3, color1 = (0.9, 0.2, 0.1), color2 = (0.1, 0.3, 0.8)),
    "white_noise_cells" => p -> white_noise(floor(p)).color,
    "magic_low" => p -> magic(p; depth = 1, scale = 1, distortion = 1),
    "brick_low" => p -> brick(p; offset = 0.4, scale = 2, bias = 0.2),
    "white_noise1" => p -> white_noise(p; dimensions = 1, w = 0.25),
)

# Baked by Blender 5.2.1's Cycles (Emission, bake type EMIT) at these points.
const CYCLES_VALUES = [
    ("noise_fbm3", (-0.034375f0, 0.928125f0, 0.37f0), (0.54068, 0.54068, 0.54068)),
    ("noise_fbm3", (0.815625f0, 1.778125f0, 0.37f0), (0.53669, 0.53669, 0.53669)),
    ("noise_fbm3_color", (-0.034375f0, 0.928125f0, 0.37f0), (0.54068, 0.41503, 0.54567)),
    ("noise_fbm3_color", (0.815625f0, 1.778125f0, 0.37f0), (0.53669, 0.56216, 0.43493)),
    ("noise_fbm3_frac", (0.815625f0, 1.778125f0, 0.37f0), (0.49503, 0.49503, 0.49503)),
    ("noise_fbm3_frac", (0.390625f0, 0.609375f0, 0.37f0), (0.50814, 0.50814, 0.50814)),
    ("noise_distort4", (-0.034375f0, 0.928125f0, 0.37f0), (0.40798, 0.43294, 0.67087)),
    ("noise_distort4", (0.815625f0, 1.778125f0, 0.37f0), (0.37138, 0.4319, 0.40363)),
    ("noise_ridged3", (-0.034375f0, 0.928125f0, 0.37f0), (1.00069, 1.00069, 1.00069)),
    ("noise_ridged3", (0.815625f0, 1.778125f0, 0.37f0), (1.50802, 1.50802, 1.50802)),
    ("noise_hetero2", (0.815625f0, 1.778125f0, 0.37f0), (0.79032, 0.79032, 0.79032)),
    ("noise_hetero2", (1.134375f0, 1.353125f0, 0.37f0), (-0.1375, -0.1375, -0.1375)),
    ("noise_multi1", (-0.034375f0, 0.928125f0, 0.37f0), (1.25, 1.25, 1.25)),
    ("noise_hybrid3_raw", (-0.034375f0, 0.928125f0, 0.37f0), (0.39236, 0.39236, 0.39236)),
    ("noise_hybrid3_raw", (0.390625f0, 0.609375f0, 0.37f0), (0.49873, 0.49873, 0.49873)),
    ("voronoi_f1", (-0.034375f0, 0.928125f0, 0.37f0), (0.60468, 0.60468, 0.60468)),
    ("voronoi_f1", (0.815625f0, 1.778125f0, 0.37f0), (0.57566, 0.57566, 0.57566)),
    ("voronoi_f2_manhattan", (0.178125f0, 1.990625f0, 0.37f0), (1.32015, 1.32015, 1.32015)),
    ("voronoi_smooth", (-0.034375f0, 0.928125f0, 0.37f0), (0.47462, 0.59253, 0.1953)),
    ("voronoi_smooth", (0.815625f0, 1.778125f0, 0.37f0), (0.82074, 0.56568, 0.44587)),
    ("voronoi_edge", (-0.034375f0, 0.928125f0, 0.37f0), (0.01613, 0.01613, 0.01613)),
    ("voronoi_edge", (0.815625f0, 1.778125f0, 0.37f0), (0.13473, 0.13473, 0.13473)),
    ("voronoi_fractal", (-0.034375f0, 0.928125f0, 0.37f0), (0.30381, 0.30381, 0.30381)),
    ("voronoi_fractal", (0.815625f0, 1.778125f0, 0.37f0), (0.37447, 0.37447, 0.37447)),
    ("voronoi_minkowski2", (-0.034375f0, 0.928125f0, 0.37f0), (0.81549, 0.81549, 0.81549)),
    ("voronoi_minkowski2", (0.390625f0, 0.609375f0, 0.37f0), (0.73959, 0.73959, 0.73959)),
    ("voronoi_4d", (-0.034375f0, 0.928125f0, 0.37f0), (0.23654, 0.96584, 0.06641)),
    ("voronoi_4d", (0.815625f0, 1.778125f0, 0.37f0), (0.61753, 0.912, 0.75434)),
    ("voronoi_1d", (-0.034375f0, 0.928125f0, 0.37f0), (0.18489, 0.18489, 0.18489)),
    ("gradient_radial", (-0.034375f0, 0.928125f0, 0.37f0), (0.75587, 0.75587, 0.75587)),
    ("gradient_radial", (0.815625f0, 1.778125f0, 0.37f0), (0.68155, 0.68155, 0.68155)),
    ("checker", (-0.034375f0, 0.928125f0, 0.37f0), (0.1, 0.3, 0.8)),
    ("white_noise_cells", (-2.79375f0, 4.13125f0, 2.0f0), (0.24955, 0.65778, 0.56136)),
    ("white_noise_cells", (1.70625f0, 8.63125f0, 2.0f0), (0.71883, 0.26316, 0.34853)),
    ("magic_low", (0.390625f0, 0.609375f0, 0.37f0), (0.23085, 0.84458, 3.0e-5)),
    ("magic_low", (1.134375f0, 1.353125f0, 0.37f0), (0.00576, 0.89619, 0.30231)),
    ("brick_low", (-0.20625f0, 0.61875f0, 0.37f0), (0.0, 0.0, 0.0)),
    ("brick_low", (0.09375f0, 0.91875f0, 0.37f0), (0.49332, 0.49332, 0.49332)),
    ("white_noise1", (-0.034375f0, 0.928125f0, 0.37f0), (0.64655, 0.64655, 0.64655)),
]

KA.@kernel function noise_dimensions!(out, points)
    i=KA.@index(Global)
    p=points[i]
    @inbounds begin
        out[1,i]=Hikari._snoise(p[1])
        out[2,i]=Hikari._snoise(Vec2f(p[1],p[2]))
        out[3,i]=Hikari._snoise(Vec3f(p[1],p[2],p[3]))
        out[4,i]=Hikari._snoise(p)
    end
end

@testset "Node textures" begin
    @testset "inputs" begin
        @test evaluate(xyz(); p = Point3f(1, -2, 3)) == Vec3f(1, -2, 3)
        @test evaluate(uv(); uv = Point2f(0.25, 0.75)) == Vec3f(0.25, 0.75, 0)
        @test evaluate(normal(); n = Vec3f(0, 1, 0)) == Vec3f(0, 1, 0)
    end

    @testset "arithmetic and swizzles" begin
        p = xyz()
        @test at(p.zx * 2 + 1, (1, 2, 3)) == Vec3f(7, 3, 1)
        @test at(p.y, (1, 2, 3)) == Vec3f(2, 2, 2)
        @test at(combine(p.z, p.x, 5), (1, 2, 3)) == Vec3f(3, 1, 5)
        @test at(norm(p.xy), (3, 4, 7)) == Vec3f(5, 5, 5)
        @test at(fract(p) - floor(-p), (1.25, 2.5, -0.75)) ≈ Vec3f(2.25, 3.5, 0.25)
        @test at(p / 0, (1, 2, 3)) == Vec3f(0, 0, 0)
        @test at(mod(p, 2), (-1, 3, 4.5)) == Vec3f(1, 1, 0.5)
        @test at(rem(p, 2), (-1, 3, 4.5)) == Vec3f(-1, 1, 0.5)
        @test at(dot(p, (1, 1, 1)) + cross(p, (0, 0, 1)), (1, 2, 3)) == Vec3f(8, 5, 6)
    end

    @testset "math operations" begin
        x = xyz().x
        @test scalar_at(math(:smooth_min, x, 1.0, 0.0), (0.5, 0, 0)) == 0.5f0
        @test scalar_at(math(:smooth_min, x, 1.0, 1.0), (1, 0, 0)) ≈ 1 - 1 / 6
        @test scalar_at(math(:wrap, x, 1.0, 0.0), (2.25, 0, 0)) ≈ 0.25f0
        @test scalar_at(math(:pingpong, x, 1.0), (1.25, 0, 0)) ≈ 0.75f0
        @test scalar_at(math(:snap, x, 0.5), (1.3, 0, 0)) == 1.0f0
        @test scalar_at(math(:compare, x, 1.0, 0.1), (1.05, 0, 0)) == 1f0
        @test scalar_at(math(:compare, x, 1.0, 0.1), (1.2, 0, 0)) == 0f0
        @test scalar_at(log(2, x), (8, 0, 0)) ≈ 3f0
        @test scalar_at(sign(x) * trunc(x), (-2.5, 0, 0)) == 2f0
        @test_throws "math operation must be one of" math(:nope, x)
    end

    @testset "vector math" begin
        p = xyz()
        @test at(vector_math(:project, p, (1, 0, 0)), (3, 4, 5)) == Vec3f(3, 0, 0)
        @test at(vector_math(:reflect, p, (0, 0, 2)), (1, 0, -1)) == Vec3f(1, 0, 1)
        @test at(vector_math(:faceforward, p, (0, 0, 1), (0, 0, 1)), (1, 2, 3)) == Vec3f(-1, -2, -3)
        @test at(vector_math(:distance, p, (1, 1, 1)), (1, 4, 5)) == Vec3f(5, 5, 5)
        @test at(vector_math(:scale, p; scale = 2), (1, 2, 3)) == Vec3f(2, 4, 6)
    end

    @testset "map range and clamp" begin
        x = xyz().x
        @test scalar_at(map_range(x, 0, 2, 10, 20), (1, 0, 0)) == 15f0
        @test scalar_at(map_range(x, 0, 2, 10, 20), (5, 0, 0)) == 20f0
        @test scalar_at(map_range(x, 0, 2, 10, 20; clamp = false), (5, 0, 0)) == 35f0
        @test scalar_at(map_range(x, 0, 1, 0, 1; interpolation = :stepped, steps = 4), (0.6, 0, 0)) == 0.75f0
        @test scalar_at(map_range(x, 0, 1, 0, 1; interpolation = :smoothstep), (0.25, 0, 0)) ≈ 0.15625f0
        @test scalar_at(map_range(x, 1, 1, 3, 4), (1, 0, 0)) == 0f0
        @test at(map_range(xyz(), 0, 2, 10, 20), (1, 2, 0)) == Vec3f(15, 20, 10)
        @test scalar_at(clamp(x, 2, 1; range = true), (5, 0, 0)) == 2f0
        @test scalar_at(smoothstep(0, 2, x), (1, 0, 0)) == 0.5f0
    end

    @testset "mix and colour" begin
        a, b = (0.2, 0.4, 0.8), (0.5, 0.5, 0.5)
        t = xyz().x
        @test at(mix(a, b, t), (0.5, 0, 0)) ≈ Vec3f(0.35, 0.45, 0.65)
        @test at(mix(a, b, t; blend = :multiply), (1, 0, 0)) ≈ Vec3f(0.1, 0.2, 0.4)
        @test at(mix(a, b, t; blend = :screen), (1, 0, 0)) ≈ Vec3f(0.6, 0.7, 0.9)
        @test at(mix(a, b, t; blend = :add, clamp_factor = true), (3, 0, 0)) ≈ Vec3f(0.7, 0.9, 1.3)
        @test at(mix(a, b, t; blend = :add, clamp_result = true), (1, 0, 0)) ≈ Vec3f(0.7, 0.9, 1.0)
        @test at(mix((0, 0, 0), (1, 1, 1), xyz()), (0.1, 0.2, 0.3)) ≈ Vec3f(0.1, 0.2, 0.3)
        red = constant((1, 0, 0))
        @test at(separate_color(red), (0, 0, 0)) == Vec3f(0, 1, 1)
        @test at(combine_color(1 / 3, 1, 1), (0, 0, 0)) ≈ Vec3f(0, 1, 0) atol = 1e-6
        @test at(separate_color(constant((0.5, 0.25, 0.25)); model = :hsl), (0, 0, 0)) ≈ Vec3f(0, 1 / 3, 0.375)
        @test at(hsv(red; hue = 0.5 + 1 / 3), (0, 0, 0)) ≈ Vec3f(0, 1, 0) atol = 1e-6
        @test at(hsv(red; saturation = 0), (0, 0, 0)) == Vec3f(1, 1, 1)
        @test at(gamma(constant((0.25, 4, -1)), 0.5), (0, 0, 0)) == Vec3f(0.5, 2, -1)
        @test at(bright_contrast(constant((0.5, 0.2, 0.0)); bright = 0.1, contrast = 1), (0, 0, 0)) ≈ Vec3f(0.6, 0, 0)
        @test at(invert(constant((0.25, 0.5, 1)); fac = 0.5), (0, 0, 0)) == Vec3f(0.5, 0.5, 0.5)
        @test scalar_at(rgb_to_bw(constant((1, 1, 1))), (0, 0, 0)) ≈ 1f0
    end

    @testset "color ramp" begin
        ramp(t; kw...) = at(color_ramp(xyz().x, 0.2 => (1, 0, 0), 0.6 => (0, 0, 1), 1.0 => (0, 1, 0); kw...), (t, 0, 0))
        @test ramp(0.0) == Vec3f(1, 0, 0)
        @test ramp(0.4) ≈ Vec3f(0.5, 0, 0.5) atol = 2e-3
        @test ramp(0.8) ≈ Vec3f(0, 0.5, 0.5) atol = 2e-3
        @test ramp(2.0) == Vec3f(0, 1, 0)
        @test ramp(0.5; interpolation = :constant) == Vec3f(1, 0, 0)
        @test ramp(0.7; interpolation = :constant) == Vec3f(0, 0, 1)
        @test ramp(0.3; interpolation = :ease)[1] > ramp(0.3)[1]
        @test at(color_ramp(xyz().x, 0 => 0, 1 => 1), (0.5, 0, 0)) ≈ Vec3f(0.5, 0.5, 0.5) atol = 2e-3
    end

    @testset "mapping and rotation" begin
        moved = mapping(xyz(); location = (1, 0, 0), rotation = (0, 0, π / 2), scale = (2, 2, 2))
        @test at(moved, (1, 0, 0)) ≈ Vec3f(1, 2, 0) atol = 1e-6
        back = mapping(moved; location = (1, 0, 0), rotation = (0, 0, π / 2), scale = (2, 2, 2), type = :texture)
        @test at(back, (0.3, -0.2, 0.7)) ≈ Vec3f(0.3, -0.2, 0.7) atol = 1e-6
        @test at(mapping(xyz(); location = (5, 5, 5), type = :vector), (1, 2, 3)) == Vec3f(1, 2, 3)
        @test at(transform(xyz(), [1 0 0 5; 0 2 0 0; 0 0 1 0; 0 0 0 1]), (1, 1, 1)) == Vec3f(6, 2, 1)
        @test at(vector_rotate(xyz(); type = :z, angle = π / 2), (1, 0, 0)) ≈ Vec3f(0, 1, 0) atol = 1e-6
        @test at(vector_rotate(xyz(); axis = (0, 0, 2), angle = π / 2, center = (1, 0, 0)), (2, 0, 0)) ≈ Vec3f(1, 1, 0) atol = 1e-6
        @test at(vector_rotate(xyz(); type = :euler_xyz, rotation = (0, 0, π / 2), invert = true), (0, 1, 0)) ≈ Vec3f(1, 0, 0) atol = 1e-6
    end

    @testset "noise" begin
        p = xyz()
        @test at(noise(p; detail = 0), (3, -2, 7)) == Vec3f(0.5, 0.5, 0.5)
        @test at(noise(p; detail = 0), (3.3, -2.1, 7.7)) != Vec3f(0.5, 0.5, 0.5)
        n = noise(p; detail = 3)
        q = (0.31, 0.77, -0.4)
        @test at(n.color, q)[1] == at(n, q)[1]
        @test at(n.color, q)[2] != at(n.color, q)[1]
        @test scalar_at(noise(p; w = 0.3, dimensions = 1), q) == scalar_at(noise(p; w = 0.3, dimensions = 1), (9, 9, 9))
        @test scalar_at(noise(p; dimensions = 2), (0.31, 0.77, -0.4)) == scalar_at(noise(p; dimensions = 2), (0.31, 0.77, 5))
        for type in (:multifractal, :hybrid_multifractal, :ridged_multifractal, :hetero_terrain), dims in 1:4
            @test isfinite(scalar_at(noise(p; type, dimensions = dims, detail = 2.5, w = 0.2), q))
        end
        @test_throws "noise type must be one of" noise(p; type = :perlin)
    end

    @testset "voronoi on a regular lattice" begin
        p = xyz()
        q = (2.3, -0.4, 5.9)
        lattice(; kw...) = voronoi(p; scale = 1, randomness = 0, kw...)
        nearest = round.(q)
        @test at(lattice(), q) ≈ Vec3f(fill(norm(q .- nearest), 3)) atol = 1e-6
        @test at(lattice().position, q) == Vec3f(nearest)
        @test scalar_at(lattice(; metric = :manhattan), q) ≈ sum(abs.(q .- nearest)) atol = 1e-6
        @test scalar_at(lattice(; metric = :chebychev), q) ≈ maximum(abs.(q .- nearest)) atol = 1e-6
        @test scalar_at(lattice(; feature = :distance_to_edge), q) ≈ 0.5 - maximum(abs.(q .- nearest)) atol = 1e-6
        @test scalar_at(lattice(; feature = :n_sphere_radius).radius, q) ≈ 0.5 atol = 1e-6
        second = sort(vec([norm(q .- (nearest .+ d)) for d in Iterators.product(-1:1, -1:1, -1:1)]))[2]
        @test scalar_at(lattice(; feature = :f2), q) ≈ second atol = 1e-6
        @test at(lattice().color, q) == at(lattice().color, nearest .+ 0.1)
        @test at(lattice().color, q) != at(lattice().color, nearest .+ 1)
        for feature in (:f1, :f2, :smooth_f1, :distance_to_edge, :n_sphere_radius), dims in 1:4
            v = voronoi(p; feature, dimensions = dims, detail = 1.5, w = 0.4)
            @test all(isfinite, at(v, q))
        end
    end

    @testset "patterns" begin
        p = xyz()
        @test scalar_at(checker(p; scale = 1).fac, (0.5, 0.5, 0.5)) == 0f0
        @test scalar_at(checker(p; scale = 1).fac, (1.5, 0.5, 0.5)) == 1f0
        @test at(checker(p; scale = 1, color1 = (1, 0, 0), color2 = (0, 0, 1)), (1.5, 0.5, 0.5)) == Vec3f(1, 0, 0)
        @test scalar_at(gradient(p).fac, (0.25, 0.9, 0)) == 0.25f0
        @test scalar_at(gradient(p; type = :spherical).fac, (0.3, 0.4, 0)) ≈ 0.499999f0
        @test scalar_at(gradient(p; type = :radial).fac, (0, 1, 0)) ≈ 0.75f0
        bricks = brick(p; scale = 1, color1 = (1, 0, 0), color2 = (1, 0, 0), mortar = (0, 0, 1), mortar_smooth = 0)
        @test at(bricks, (0.5, 0.125, 0)) == Vec3f(1, 0, 0)
        @test at(bricks, (0.5, 0.01, 0)) == Vec3f(0, 0, 1)
        @test scalar_at(bricks.fac, (0.5, 0.01, 0)) == 1f0
        @test scalar_at(wave(p; scale = 1, profile = :saw), (0.05π, 0, 0)) ≈ 0.5f0 atol = 1e-4
        @test scalar_at(wave(p; scale = 1, type = :rings, direction = :spherical, profile = :triangle), (0.05π, 0, 0)) ≈ 1f0 atol = 1e-4
        @test at(magic(p; depth = 0, distortion = 0), (0, 0, 0)) == Vec3f(0.5, -0.5, 1.5)
        cell(x) = at(white_noise(floor(p)), x)
        @test cell((0.2, 0.3, 0.4)) == cell((0.9, 0.1, 0.6))
        @test cell((0.2, 0.3, 0.4)) != cell((1.2, 0.3, 0.4))
        @test scalar_at(white_noise(p).value, (1, 2, 3)) == at(white_noise(p).color, (1, 2, 3))[1]
    end

    @testset "matches Cycles: $name at $p" for (name, p, value) in CYCLES_VALUES
        @test at(CYCLES_GRAPHS[name](xyz()), p) ≈ Vec3f(value) atol = 3e-3
    end

    @testset "image at computed coordinates" begin
        img = [RGBSpectrum(Float32(i), Float32(j), 0f0) for i in 1:2, j in 1:2]
        @test at(image_texture(img, xyz().xy), (1, 2, 0)) == Vec3f(1.5, 1.5, 0)
        @test evaluate(image_texture(img); uv = Point2f(0.25, 0.75)) == Vec3f(1, 1, 0)
    end

    @testset "scalar parameters read the first component" begin
        store = Raycore.MultiTypeSet(CPU())
        handle = device_param(store, xyz().z * 2)
        @test handle.kind == TexKind.NODES
        @test eval_handle(host_static(store), handle, TextureFilterContext(Point2f(0), Point3f(0, 0, 0.25))) == 0.5f0
    end

    @testset "stack slots are reused" begin
        p = xyz()
        terms = [sin(p.x * k) for k in 1:40]
        x = (0.3, 0, 0)
        @test scalar_at(reduce(+, terms), x) ≈ sum(sin(0.3f0 * k) for k in 1:40) atol = 1e-5
        @test scalar_at(foldr(+, terms), x) ≈ sum(sin(0.3f0 * k) for k in 1:40) atol = 1e-5
        wide = [sin(p * k) for k in 1:60]
        @test_throws "more than $(Hikari.NODE_STACK) floats" Hikari.compile_nodes(reduce(+, wide) + reduce(+, [t * t for t in wide]))
        shared = noise(p; detail = 1)
        @test at(shared.color + shared, x) ≈ at(shared.color, x) + at(shared, x)
    end

    @testset "programs carry the node families and stack they use" begin
        features(g) = typeof(Hikari.compile_nodes(g)).parameters[1]
        p = xyz()
        F = Hikari.NodeFeature
        families = ~(UInt32(0x3) << Hikari.STACK_SHIFT | F.TEXTURE)
        @test features(p.x * 2 + 1) & families == 0
        @test features(sin(p.x) * 2) & families == F.MATH_TRIG
        # A texture's program says so, a shader's never does: a texture read runs only the former.
        @test features(p.x * 2 + 1) & F.TEXTURE == F.TEXTURE
        shader = Hikari.ShaderGraph(Hikari.shader_node(Hikari.Principled(base_color = noise(p).color)))
        @test typeof(Hikari.compile_shader(shader.closures)).parameters[1] & F.TEXTURE == 0
        @test features(floor(p) + sqrt(p)) & (F.MATH_ROUND | F.MATH_POWER | F.MATH_TRIG) == F.MATH_ROUND | F.MATH_POWER
        @test features(noise(p)) & (F.NOISE | F.NOISE_DIMS | F.NOISE_TYPES | F.VORONOI) == F.NOISE
        @test features(noise(p; dimensions = 4, type = :ridged_multifractal)) & (F.NOISE_DIMS | F.NOISE_TYPES) == F.NOISE_DIMS | F.NOISE_TYPES
        @test features(voronoi(p; feature = :smooth_f1)) & (F.VORONOI | F.VORONOI_SMOOTH | F.VORONOI_EDGE) == F.VORONOI | F.VORONOI_SMOOTH
        @test features(mix((1, 0, 0), (0, 0, 1), p.x; blend = :screen)) & (F.MIX | F.BLEND) == F.MIX | F.BLEND
        @test Hikari._stack_size(features(sin(p.x))) == 16
        terms = [sin(p * k) for k in 1:12]
        @test Hikari._stack_size(features(reduce(+, terms) + reduce(+, [t * t for t in terms]))) == 64
    end

    @testset "alpha does not run a node program" begin
        store = Raycore.MultiTypeSet(CPU())
        handle = device_param(store, xyz().x * 0)
        @test Hikari.eval_handle_alpha(host_static(store), handle, TextureFilterContext(Point2f(0), Point3f(1, 0, 0))) == 1f0
    end

    @testset "bump mapping moves the position" begin
        store = Raycore.MultiTypeSet(CPU())
        slope = 0.5f0
        height = device_param(store, xyz().x * slope)
        tfc = TextureFilterContext(Point2f(0.3, 0.3), 0.01f0, 0f0, 0f0, 0.01f0, Point3f(0.3, 0.3, 0), Vec3f(0, 0, 1))
        ns, _ = Hikari.perturb_bump_frame(height, host_static(store), Vec3f(0, 0, 1),
                                          Vec3f(1, 0, 0), Vec3f(0, 1, 0), Vec3f(0), Vec3f(0), Vec3f(0, 0, 1), tfc)
        @test ns ≈ normalize(Vec3f(-slope, 0, 1))
    end

    @testset "updating a graph reuses its slot" begin
        store = Raycore.MultiTypeSet(CPU())
        handle = device_param(store, xyz().x * 2)
        updated = Raycore.update_item(store, handle, xyz().x * 3)
        @test updated == handle
        tfc = TextureFilterContext(Point2f(0), Point3f(1, 0, 0))
        @test eval_handle(host_static(store), updated, tfc) == 3f0
        # A longer program of the same node families is resized where it is.
        longer = Raycore.update_item(store, handle, xyz().x * 3 + 1)
        @test longer == handle
        @test eval_handle(host_static(store), longer, tfc) == 4f0
    end

    # An update compiled the graph again, and compiling stored every image it read: each update
    # took a new array per image, kept the one it replaced, and changed the store's layout, which
    # drops every plan recorded against it. The images go where the old program's are.
    @testset "updating a graph keeps the images it reads where they are" begin
        store = Raycore.MultiTypeSet(CPU())
        image(c, n = 4) = fill(RGBSpectrum(c...), n, n)
        handle = device_param(store, image_texture(image((1, 0, 0))) * 0.5)
        arrays, layout = length(store.texture_gpu_arrays), store.layout
        rgb(h) = (s = eval_handle_spectrum(host_static(store), h, TextureFilterContext(Point2f(0.5), Point3f(0), Vec3f(0, 0, 1)));
                  Vec3f(s.c[1], s.c[2], s.c[3]))
        updated = Raycore.update_item(store, handle, image_texture(image((0, 0, 1))) * 0.25)
        @test updated == handle
        @test (length(store.texture_gpu_arrays), store.layout) == (arrays, layout)
        @test rgb(updated) == Vec3f(0, 0, 0.25)
        # Another size is the same array, resized.
        resized = Raycore.update_item(store, handle, image_texture(image((0, 1, 0), 8)) * 0.25)
        @test resized == handle
        @test length(store.texture_gpu_arrays) == arrays
        @test rgb(resized) == Vec3f(0, 0.25, 0)
        # One image read twice is one array, and stays one.
        once = device_param(store, (img = image_texture(image((1, 1, 0))); img * img))
        arrays = length(store.texture_gpu_arrays)
        again = Raycore.update_item(store, once, (img = image_texture(image((0, 1, 1))); img * img))
        @test again == once
        @test length(store.texture_gpu_arrays) == arrays
        @test rgb(again) == Vec3f(0, 1, 1)
    end

    @testset "rendered by position ($(hw ? "hardware" : "software") tracing)" for hw in (false, true)
        backend = Mantle.defaultbackend()
        scene = Hikari.Scene(; backend, hw_accel = hw)
        sphere = GeometryBasics.normal_mesh(GeometryBasics.Tesselation(GeometryBasics.Sphere(Point3f(0), 1f0), 32))
        push!(scene, sphere, Hikari.Diffuse(color = mix((0.8, 0.05, 0.05), (0.05, 0.05, 0.8), greater(xyz().x, 0))))
        push!(scene, Hikari.PointLight(Point3f(0, 0.5, 4), RGBSpectrum(40f0)))
        Hikari.sync!(scene)
        film = Hikari.Film(backend, Hikari.Film(Point2f(32, 32)))
        camera = Hikari.PerspectiveCamera(Point3f(0, 0, 4), Point3f(0), film; fov = 40f0)
        vp = Hikari.VolPath(samples = 4, max_depth = 2, hw_accel = hw)
        vp(scene, film, camera)
        img = Array(film.framebuffer)
        close(vp)
        channels(cols) = (r = mean(p.r for p in img[12:20, cols]), b = mean(p.b for p in img[12:20, cols]))
        negative_x, positive_x = channels(6:12), channels(21:27)
        @test negative_x.r > 10 * negative_x.b
        @test positive_x.b > 10 * positive_x.r
    end
    @testset "noise dimensions agree on host and device" begin
        points=[Vec4f(-3.17,0.23,9.71,0.31),Vec4f(0.27,0.73,-0.19,4.37),
                Vec4f(100002.3,-200001.7,0.13,3.71),Vec4f(-1000003.4,1.2,3.5,5.7)]
        expected=hcat([Float32[Hikari._snoise(p[1]), Hikari._snoise(Vec2f(p[1],p[2])),
                              Hikari._snoise(Vec3f(p[1],p[2],p[3])),Hikari._snoise(p)] for p in points]...)
        backend=Mantle.defaultbackend()
        device_points=KA.allocate(backend,Vec4f,length(points)); copyto!(device_points,points)
        out=KA.zeros(backend,Float32,4,length(points))
        noise_dimensions!(backend)(out,device_points;ndrange=length(points))
        Mantle.waitidle(Mantle.Device(backend))
        got=Array(out)
        @test maximum(abs,expected)>0.05f0
        @test got ≈ expected atol=2f-6 rtol=2f-5
    end

end
