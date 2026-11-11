using Test, Hikari, Raycore, KernelAbstractions, GeometryBasics, LinearAlgebra
using Hikari.TextureNodes
using Mantle

function read_packed_principled!(out, values, program, start, sc, capacity, hit)
    KernelInterface.get_global_id().x == 1 || return nothing
    gathered = Hikari.GatheredStack(values, capacity, hit)
    @inbounds out[1] = Hikari.principled_inputs(gathered, program, start, sc)
    return nothing
end

function check_device_principled(expected, inputs, packed, start, sc)
    dev = Mantle.Device(Mantle.defaultbackend())
    arrays = Any[]
    try
        # Poison every field so an unwritten output cannot pass the comparison.
        poison = reinterpret(Hikari.PrincipledInputs,
            fill(NaN32, sizeof(Hikari.PrincipledInputs) ÷ sizeof(Float32)))
        out = Mantle.devicearray(dev, collect(poison)); push!(arrays, out)
        values = Mantle.devicearray(dev, inputs.values); push!(arrays, values)
        program = Mantle.devicearray(dev, collect(packed)); push!(arrays, program)
        KernelInterface.@launch dev ndrange=1 read_packed_principled!(
            out, values, program, start, sc, inputs.capacity, Int32(2))
        Mantle.waitidle(dev)
        actual = only(Array(out))
        for name in fieldnames(Hikari.PrincipledInputs)
            @test isequal(getfield(actual, name), getfield(expected, name))
        end
    finally
        foreach(Mantle.unsafe_free!, reverse(arrays))
    end
    return nothing
end

# A program whose `pointer` is an address-space `LLVMPtr`, as a Metal device array's is.
# The packed record was read through `Ptr{UInt8}(pointer(prog, i))`, which has no
# method for one, so every Principled shading kernel kept a `MethodError` on Metal.
struct LLVMPtrProgram{T, V <: AbstractVector{T}} <: AbstractVector{T}
    data::V
end
Base.size(p::LLVMPtrProgram) = size(p.data)
Base.getindex(p::LLVMPtrProgram, i::Integer) = p.data[i]
Base.pointer(p::LLVMPtrProgram{T}, i::Integer) where {T} = reinterpret(Core.LLVMPtr{T, 0}, pointer(p.data, i))

@testset "packed Principled inputs preserve constants, gathered values and geometry" begin
    signal = noise(xyz())
    descriptions = (
        Hikari.Principled(),
        Hikari.Principled(emission_color = (0.2, 0.3, 0.8), emission_strength = signal.fac),
        Hikari.Principled(emission_color = signal.color, emission_strength = 0),
        Hikari.Principled(emission_color = signal.fac, emission_strength = signal.fac),
        Hikari.Principled(base_color = signal.color, roughness = signal.fac,
            normal = bump(signal.fac), coat_weight = 0.4, transmission_weight = 0.3,
            emission_color = signal.color, emission_strength = signal.fac),
        Hikari.Principled(base_color = signal.fac, coat_tint = signal.fac,
            subsurface_radius = signal.color, subsurface_scale = signal.fac,
            thin_film_thickness = signal.fac * 400),
        Hikari.Mix(Hikari.Principled(base_color = signal.color),
            Hikari.Principled(emission_strength = signal.fac); factor = wave(xyz()).fac),
        Hikari.Add(Hikari.Principled(roughness = signal.fac), Hikari.Sheen(color = signal.color)),
    )
    for description in descriptions, normal in (Vec3f(0, 0, 1), normalize(Vec3f(0.2, -0.3, 1)))
        store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
        material = Hikari.to_device_material(store, Hikari.surface_material(description))
        textures = Raycore.StaticMultiTypeSet((), Tuple(T[a for a in store.texture_gpu_arrays if typeof(a) === T] for T in store.texture_order))
        sc = Hikari.ShaderContext(Hikari.TextureFilterContext(Point2f(0.3, 0.6), Point3f(0.3, -0.2, 0.7), normal),
            normalize(Vec3f(0.2, 0.1, 1)), Vec3f(1f-3, 0, 0), Vec3f(0, 1f-3, 0), Vec3f(1, 0, 0),
            normal, 1f0, Vec3f(0), Point2f(0))
        original = Hikari.shader_programs(textures, material.program)[material.program.idx]
        values, original_i = Hikari.run_values(original, textures, sc)
        capacity, hit = Int32(3), Int32(2)
        inputs = Hikari.ShaderInputs(fill(NaN32, max(Int(material.inputs), 1) * capacity), zeros(Int32, capacity), capacity)
        Hikari.gather_inputs!(inputs, hit, values, original, original_i)
        @test count(!isnan, inputs.values) == material.inputs
        gathered = Hikari.GatheredStack(inputs.values, capacity, hit)
        packed = Hikari.closure_programs(textures)[material.closures]
        packed_i = Int32(1)
        while original[original_i].op != Hikari.NodeOp.RESULT
            ins = original[original_i]
            if ins.op == Hikari.NodeOp.PRINCIPLED
                @test packed[packed_i].out[1] == Hikari.PACKED_PRINCIPLED_INPUTS
                expected = Hikari.principled_inputs(values, original, original_i, sc)
                actual = @inferred Hikari.principled_inputs(gathered, packed, packed_i, sc)
                through_llvmptr = GC.@preserve packed Hikari.principled_inputs(gathered, LLVMPtrProgram(packed), packed_i, sc)
                for name in fieldnames(Hikari.PrincipledInputs)
                    @test isequal(getfield(actual, name), getfield(expected, name))
                    @test isequal(getfield(through_llvmptr, name), getfield(expected, name))
                end
                check_device_principled(expected, inputs, packed, packed_i, sc)
            end
            chunks = Hikari.surface_chunks(ins)
            original_i = Hikari.next_node(original, original_i, chunks)
            packed_i = Hikari.next_node(packed, packed_i, chunks)
        end
        @test packed[packed_i].op == Hikari.NodeOp.RESULT
    end
end
