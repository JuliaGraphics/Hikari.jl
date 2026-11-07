using Test, Hikari, GeometryBasics, Mantle, Adapt, Random
using LinearAlgebra: dot, norm
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index

@kernel function nanofluid_samples!(out, tree, points)
    i = @index(Global)
    out[i] = Hikari.nanosample(tree, tree.words, points[i])
end

@kernel function boundary_offsets!(out,points,normals,directions)
    i=@index(Global)
    out[i]=Hikari.offset_surface_origin(points[i],normals[i],directions[i])
end

@testset "Ray origins leave boundaries at every coordinate scale" begin
    points=Point3f[]; normals=Vec3f[]; directions=Vec3f[]
    for coordinate in (-3000f0,-10f0,-1f-4,0f0,1f-4,10f0,3000f0), axis in 1:3, sign in (-1f0,1f0)
        p=Point3f(ntuple(a->a==axis ? coordinate : 0f0,3))
        n=Vec3f(ntuple(a->a==axis ? 1f0 : 0f0,3)); d=n*sign
        q=Hikari.offset_surface_origin(p,n,d)
        @test all(isfinite,q)
        @test dot(q-p,d)>0f0
        @test norm(q-p)<max(2f-4,abs(coordinate)*2f-6)
        push!(points,p); push!(normals,n); push!(directions,d)
    end
    backend=Mantle.defaultbackend()
    result=KA.allocate(backend,Point3f,length(points))
    boundary_offsets!(backend,64)(result,Mantle.devicearray(backend,points),
        Mantle.devicearray(backend,normals),Mantle.devicearray(backend,directions);ndrange=length(points))
    @test Array(result) == Hikari.offset_surface_origin.(points,normals,directions)
end

@testset "Thermal NanoVDB source" begin
    for temperature in (300f0, 1000f0, 2000f0, 6000f0), λ in (400f0, 550f0, 700f0)
        expected = Hikari.blackbody(Float64(λ), Float64(temperature)) * 1e-9
        @test Hikari.thermalradiance(λ, temperature) ≈ expected rtol=2e-5 atol=1e-30
    end
    @test Hikari.thermalradiance(550f0, 0f0) == 0f0
    @test Hikari.thermalradiance(550f0, -100f0) == 0f0
    @test Hikari.thermalradiance(550f0, 2000f0) > Hikari.thermalradiance(550f0, 1000f0)
end

@testset "Transmission traverses the entire fine grid" begin
    grid = Hikari.MajorantGrid(Vec{3,Int32}(1024, 1, 1))
    fill!(grid.voxels, 0.001f0)
    bounds = Hikari.Bounds3(Point3f(0), Point3f(1024, 1, 1))
    origin = Point3f(0, 0.5, 0.5); direction = Vec3f(1, 0, 0)
    iter = Hikari.RayMajorantIterator(Hikari.create_dda_iterator(grid, bounds, origin, direction,
              0f0, 1024f0, Hikari.SpectralRadiance(1f0)))
    control = Hikari.MediumProperties(Hikari.SpectralRadiance(0.001f0), Hikari.SpectralRadiance(0f0),
                                      Hikari.SpectralRadiance(0f0), 0f0)
    transmittance, _, _ = Hikari.ratio_tracking_dda(iter, origin, direction, nothing,
        Hikari.SetKey(), nothing, Hikari.sample_wavelengths_visible(0.5f0), control, 0.001f0)
    @test transmittance[1] ≈ exp(-1.024f0) rtol=1e-4
end

@testset "Majorant intervals retain heterogeneous extinction" begin
    grid = Hikari.MajorantGrid(Float32[0,0,1,1,0], Vec{3,Int32}(5,1,1))
    bounds = Hikari.Bounds3(Point3f(0),Point3f(5,1,1))
    iterator = Hikari.create_dda_iterator(grid,bounds,Point3f(0,.5,.5),Vec3f(1,0,0),0f0,5f0,
                                         Hikari.SpectralRadiance(1f0))
    optical_depth = 0f0; length_covered = 0f0
    while true
        segment, iterator, valid = Hikari.dda_next(iterator,nothing)
        valid || break
        distance = segment.t_max-segment.t_min
        optical_depth += distance*segment.σ_maj[1]
        length_covered += distance
    end
    @test optical_depth ≈ 2f0
    @test length_covered ≈ 5f0
end

@testset "NanoVDB fluid fields" begin
    backend = Mantle.defaultbackend()
    origins = Vec{3,Int32}[Vec{3,Int32}(8i, 8j, 8k) for i in 0:1 for j in 0:1 for k in 0:1]
    # Unequal adjacent leaves exercise both the cached lookup and interpolation
    # across leaves. The one-cell majorant must be max(values), not their sum.
    values = vcat([fill(Float32(l / 8), 512) for l in 1:8]...)
    tree = Hikari.NanoTree(Mantle.devicearray(backend, origins), Mantle.devicearray(backend, values),
                          Point3f(0), 0.1f0; backend)
    host = Hikari.NanoTree(Array(tree.words), tree.upper, tree.lower, tree.leaf, tree.uppers,
                          tree.lowers, tree.leaves, tree.roottiles, tree.lo, tree.voxel,
                          tree.background, tree.imin, tree.imax)
    bounds = Hikari.Bounds3(Point3f(-0.2), Point3f(1.8))
    one = Hikari.nanomajorant(tree, bounds, Vec{3,Int32}(1, 1, 1))
    @test one.voxels == [1f0]
    grid = Hikari.nanomajorant(tree, bounds, Vec{3,Int32}(11, 9, 7))
    rng = MersenneTwister(21020)
    points = [Point3f(-0.2f0 .+ 2f0 .* rand(rng, Float32, 3)) for _ in 1:1000]
    append!(points, [Hikari.worldspace(host, Vec3f(x, y, z)) for x in (6.999f0, 7f0, 7.5f0, 8f0)
                    for y in (7f0, 7.5f0, 8f0) for z in (7f0, 7.5f0, 8f0)])
    expected = [Hikari.nanosample(host, host.words, p) for p in points]
    actual = KA.allocate(backend, Float32, length(points))
    nanofluid_samples!(backend, 256)(actual, tree, Mantle.devicearray(backend, points); ndrange=length(points))
    @test maximum(abs.(Array(actual) .- expected)) <= 2f-6
    for (p, value) in zip(points, expected)
        c = clamp.(floor.(Int32, (p - bounds.p_min) ./ (bounds.p_max - bounds.p_min) .* grid.res),
                   Int32(0), grid.res .- Int32(1))
        idx = c[1] + grid.res[1] * (c[2] + grid.res[2] * c[3]) + 1
        @test 0f0 <= value <= grid.voxels[idx] + 2f-6
    end
    water = Hikari.WaterMedium(nothing; air=((tree, 0.001f0, 0.85f0), (tree, 0.002f0, 0.5f0)),
                               bounds, cell=2f0, ior=1f0, backend)
    @test Hikari.refracting(water) == 0f0
    refractive_water = Hikari.WaterMedium(nothing; ior=1.34f0,bounds,cell=2f0,backend)
    @test Hikari.refracting(refractive_water) == 1.34f0
    @test only(water.majorant_grid.voxels) ≈ (1 - 0.85f0) * 1.5f0 / 0.001f0 +
                                             (1 - 0.5f0) * 1.5f0 / 0.002f0
    mktempdir() do dir
        path = joinpath(dir, "fluid.nvdb")
        Hikari.save_nanovdb(path, tree)
        buffer, _ = Hikari.parse_nanovdb_buffer(path)
        @test reinterpret(UInt64, buffer[9:16])[1] == typemax(UInt64)
        @test reinterpret(Float64, buffer[Hikari.MAP_OFFSET+88:Hikari.MAP_OFFSET+95])[1] ≈ Float64(tree.voxel)
        native = Hikari.NanoVDBMedium(path; majorant_res=Vec3i(4, 4, 4))
        @test maximum(abs.([Hikari.sample_nanovdb_density(native, nothing, p) for p in points] .- expected)) <= 2f-6
        Hikari.save_nanovdb(path, tree; name="temperature", gridclass=0)
        buffer, _ = Hikari.parse_nanovdb_buffer(path)
        @test String(buffer[41:51]) == "temperature"
        @test reinterpret(UInt32, buffer[633:636])[1] == 0
    end
    temp = Hikari.NanoTree(Mantle.devicearray(backend, origins), Mantle.devicearray(backend, fill(1800f0, 4096)),
                          Point3f(0), 0.1f0; backend)
    temphost = Hikari.NanoTree(Array(temp.words), temp.upper, temp.lower, temp.leaf, temp.uppers,
                              temp.lowers, temp.leaves, temp.roottiles, temp.lo, temp.voxel,
                              temp.background, temp.imin, temp.imax)
    gas = Hikari.NanoGridMedium(host; temperature=temphost, σ_a=Hikari.RGBSpectrum(1f0),
                                σ_s=Hikari.RGBSpectrum(0f0), majorant_res=(4, 4, 4))
    # A localized peak must not inflate the rest of its brick; its trilinear
    # interpolation must still remain below every local bound.
    impulse = zeros(Float32,512); impulse[1+64*3+8*3+3] = 1f0
    peak = Hikari.NanoTree(Mantle.devicearray(backend, [Vec{3,Int32}(0)]),
                          Mantle.devicearray(backend,impulse),Point3f(0),0.1f0;backend)
    peakhost = Hikari.NanoTree(Array(peak.words),peak.upper,peak.lower,peak.leaf,peak.uppers,
                              peak.lowers,peak.leaves,peak.roottiles,peak.lo,peak.voxel,
                              peak.background,peak.imin,peak.imax)
    tight = Hikari.nanomajorant(peak,bounds,Vec{3,Int32}(32))
    loose = Hikari.nanomajorant(peak,bounds,Vec{3,Int32}(32);tight=false)
    @test sum(tight.voxels) < sum(loose.voxels)/10
    for i in 2:0.2:4, j in 2:0.2:4, k in 2:0.2:4
        p = Hikari.worldspace(peakhost,Vec3f(i,j,k))
        cell = clamp.(floor.(Int32,(p-bounds.p_min)./(bounds.p_max-bounds.p_min).*32),Int32(0),Int32(31))
        @test Hikari.nanosample(peakhost,peakhost.words,p) <=
            Hikari.majorant_lookup(tight,nothing,cell[1],cell[2],cell[3])+1f-6
    end
    λ = Hikari.sample_wavelengths_visible(0.5f0)
    mp = Hikari.sample_point(gas, nothing, Hikari.get_rgb2spec_table(), Point3f(0.4), λ)
    @test Hikari.is_emissive(gas)
    @test !Hikari.is_emissive(Hikari.NanoGridMedium(host; majorant_res=(1, 1, 1)))
    @test mp.Le[1] ≈ Hikari.thermalradiance(λ.lambda[1], 1800f0)
    @test mp.σ_a[1] > 0f0
    @test mp.σ_s[1] == 0f0
end

@testset "Seawater spectra" begin
    # Measured Pope & Fry data anchors, in inverse metres.
    @test Hikari.purewater_absorption(440) ≈ 0.00635 rtol=1e-5
    @test Hikari.purewater_absorption(550) ≈ 0.0565 rtol=1e-5
    @test Hikari.purewater_absorption(700) ≈ 0.624 rtol=1e-5
    pure = Hikari.Seawater(; chlorophyll=0, minerals=0, cdom=0)
    for λ in 360:5:830
        @test Hikari.absorption(pure, λ) ≈ Hikari.purewater_absorption(λ)
        @test Hikari.scattering(pure, λ) > 0
        @test isfinite(Hikari.backscattering(pure, λ))
        @test Hikari.backscattering(pure, λ) ≈ 0.5Hikari.scattering(pure, λ)
    end
    bio = Hikari.Seawater(; chlorophyll=0.1, minerals=0, cdom=0)
    bp550 = 0.416 * 0.1^0.766
    for λ in (400, 550, 700)
        molecular = Hikari.seawater_scattering(λ; temperature=bio.temperature, salinity=bio.salinity)
        @test Hikari.scattering(bio, λ) - molecular ≈ bp550 * 550 / λ
        expected = bp550 * (0.002 + 0.01 * (0.5 - 0.25log10(0.1)) * (λ / 550)^(0.5 * (log10(0.1) - 0.3)))
        @test Hikari.backscattering(bio, λ) - 0.5molecular ≈ expected
    end
end
