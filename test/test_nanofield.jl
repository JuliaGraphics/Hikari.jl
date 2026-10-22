using Test, Hikari, GeometryBasics, Mantle, Random
import KernelAbstractions as KA
using KernelAbstractions: @kernel, @index
import Hikari.Raycore as RC
import Hikari.Adapt as Adapt

@kernel function nanofield_samples!(out, field, points)
    i = @index(Global)
    out[i] = Hikari.nanofield_sample(field, nothing, points[i])
end

@testset "Surface file nodes decode on demand and reject changed revisions" begin
    mktempdir() do directory
        path=joinpath(directory,"surface.nvdb")
        tree=Hikari.NanoTree([Vec{3,Int32}(0)],ones(Float32,512),Point3f(0),.1f0;backend=KA.CPU())
        Hikari.save_nanovdb(path,tree)
        input=Hikari.defer_nanotree(path)
        @test input.tree===nothing
        @test read(input).words==tree.words
        @test read(input)===Hikari.resolve_nanotree(input)
        @test Hikari.resolve_nanotree(tree)===tree
        open(io->write(io,UInt8(0)),path,"a")
        @test_throws ErrorException read(input)
        rm(path)
        @test_throws ErrorException read(input)
        @test_throws ErrorException Hikari.defer_nanotree(path)
    end
end

@kernel function nanofield_stored_samples!(out, media, field, points)
    i = @index(Global)
    out[i] = Hikari.nanofield_sample(field, media, points[i])
end

function test_nanofield(; backend=nothing)
    @testset "File-backed NanoVDB fields and lazy world-space blending" begin
        rng = MersenneTwister(20261021)
        origins = Vec{3,Int32}[Vec{3,Int32}(-8,0,0),Vec{3,Int32}(8,0,0)]
        values = rand(rng,Float32,1024)
        a = Hikari.NanoTree(origins,values,Point3f(-.23,.17,-.51),.1f0;
            background=.125f0,backend=KA.CPU())
        b = Hikari.NanoTree(origins,2f0 .* values,Point3f(.41,-.13,.07),.15f0;
            background=.25f0,backend=KA.CPU())
        points = [Point3f(-2f0 .+ 5f0 .* rand(rng,Float32,3)) for _ in 1:1000]
        mktempdir() do dir
            path = joinpath(dir,"density.nvdb")
            Hikari.save_nanovdb(path,a)
            loaded = Hikari.load_nanotree(path)
            @test loaded.words == a.words
            @test loaded.lo ≈ a.lo atol=1f-7
            @test loaded.voxel == a.voxel
            @test (loaded.imin,loaded.imax) == (a.imin,a.imax)
            for p in points
                @test Hikari.nanofield_sample(loaded,nothing,p) ≈
                    Hikari.nanofield_sample(a,nothing,p) atol=2f-6
            end
            if backend !== nothing
                uploaded = Hikari.load_nanotree(path;backend)
                @test Array(uploaded.words) == a.words
            end
            original = read(path)
            write(path,original[1:end-3])
            @test_throws ErrorException Hikari.load_nanotree(path)
            write(path,original)
        end
        @test_throws ArgumentError Hikari.NanoBlend(a,b,NaN)
        @test_throws ArgumentError Hikari.NanoBlend(a,b,-.1)
        @test_throws ArgumentError Hikari.NanoBlend(a,b,1.1)
        bounds = Hikari.nanobounds(Hikari.NanoBlend(a,b,.5))
        res = Vec{3,Int32}(7,8,9)
        bound = Hikari.nanomajorant(Hikari.NanoBlend(a,b,.5),bounds,res)
        for weight in (0f0,.1f0,.5f0,.9f0,1f0)
            field = Hikari.NanoBlend(a,b,weight)
            @test field.a === a && field.b === b
            gas = Hikari.NanoGridMedium(field;majorant_res=Tuple(res))
            @test gas.majorant_grid.voxels == bound.voxels
            for p in points
                expected = (1f0-weight)*Hikari.nanosample(a,a.words,p) +
                    weight*Hikari.nanosample(b,b.words,p)
                actual = Hikari.nanofield_sample(field,nothing,p)
                @test actual ≈ expected atol=2f-6
                if all(bounds.p_min .<= p .<= bounds.p_max)
                    cell = clamp.(floor.(Int32,(p-bounds.p_min)./(bounds.p_max-bounds.p_min).*res),
                        Int32(0),res .- Int32(1))
                    index = cell[1] + res[1]*(cell[2]+res[2]*cell[3])+1
                    @test actual <= bound.voxels[index]+2f-6
                end
            end
            if backend !== nothing
                upload(t) = Hikari.NanoTree(Mantle.devicearray(backend,t.words),t.upper,t.lower,t.leaf,
                    t.uppers,t.lowers,t.leaves,t.roottiles,t.lo,t.voxel,t.background,t.imin,t.imax)
                device_field = Hikari.NanoBlend(upload(a),upload(b),weight)
                output = KA.allocate(backend,Float32,length(points))
                nanofield_samples!(backend,64)(output,device_field,Mantle.devicearray(backend,points);
                    ndrange=length(points))
                reference = Hikari.nanofield_sample.(Ref(field),nothing,points)
                max_error = maximum(abs.(Array(output) .- reference))
                println((nano_blend_weight=weight, max_device_error=max_error))
                @test max_error <= 2f-6
            end
        end
        if backend !== nothing
            set = RC.MultiTypeSet(backend)
            medium = Hikari.NanoGridMedium(Hikari.NanoBlend(a,b,0f0);majorant_res=Tuple(res))
            key = push!(set,medium)
            stored = set[key]
            storage = copy(set.texture_gpu_arrays)
            output = KA.allocate(backend,Float32,length(points))
            device_points = Mantle.devicearray(backend,points)
            for weight in (.5f0,1f0,0f0,.25f0,.25f0)
                RC.update!(set,key,Hikari.with_blend_weight(stored,weight))
                current = set[key]
                @test current.density.weight == weight
                @test length(set.texture_gpu_arrays) == length(storage)
                @test all(a === b for (a,b) in zip(storage,set.texture_gpu_arrays))
                @test current.majorant_grid === stored.majorant_grid
                nanofield_stored_samples!(backend,64)(output,Adapt.adapt(backend,set.static),
                    current.density,device_points;ndrange=length(points))
                expected = Hikari.nanofield_sample.(Ref(Hikari.NanoBlend(a,b,weight)),nothing,points)
                @test maximum(abs.(Array(output) .- expected)) <= 2f-6
            end
        end
    end
end

test_nanofield()

@testset "NanoVDB ZIP decompression validates complete streams" begin
    data = UInt8[1,2,3,4,5]
    compressed = Hikari.compress_zlib(data)
    @test Hikari.decompress_nanovdb(compressed) == data
    @test Hikari.decompress_nanovdb(compressed;expected_size=length(data)) == data
    @test_throws ErrorException Hikari.decompress_nanovdb(compressed[1:end-2])
    @test_throws ErrorException Hikari.decompress_nanovdb(compressed;expected_size=4)
    @test_throws ErrorException Hikari.decompress_nanovdb(compressed;expected_size=6)
    # The former fixed200MB buffer silently returned a truncated field here.
    large = zeros(UInt8,200_000_017)
    large[end] = 0xa7
    Hikari.compress_zlib(UInt8[0])
    packed = Ref{Vector{UInt8}}()
    allocated = @allocated packed[] = Hikari.compress_zlib(large)
    @test allocated < 32*1024*1024
    restored = Hikari.decompress_nanovdb(packed[];expected_size=length(large))
    @test restored == large
end

@testset "ZIP compression spans arbitrary input and output chunk boundaries" begin
    for n in (0,1,4095,4096,4097,100_003)
        original=rand(MersenneTwister(n),UInt8,n)
        for chunk in (1,257,4096)
            io=IOBuffer()
            count=Hikari.compress_zlib!(io,
                (view(original,1:n÷2),view(original,n÷2+1:n));chunk_bytes=chunk)
            compressed=take!(io)
            @test count==length(compressed)
            @test Hikari.decompress_nanovdb(compressed)==original
        end
    end
end
