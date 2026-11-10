module DecodedInputsProbe
using Hikari, Mantle, JSON3, Statistics, Test
using ..MaterialComparison

@generated function read_record(prog,i::Int32,sc::Hikari.ShaderContext)
    fields = map(fieldnames(Hikari.PrincipledInputs)) do name
        name === :n && return :((flags & 0x01) != 0 ? sc.tfc.n : p.n)
        name === :coat_n && return :((flags & 0x02) != 0 ? sc.tfc.n : p.coat_n)
        name === :tangent && return :((flags & 0x04) != 0 ? sc.dpdus : p.tangent)
        return :(getfield(p,$(QuoteNode(name))))
    end
    offset = fieldoffset(Hikari.NodeInstr{Hikari.NodeFeature.CLOSURES},7)
    return quote
        $(Expr(:meta,:inline))
        @inbounds ins = prog[i]
        flags = ins.out[2]
        p = unsafe_load(Ptr{Hikari.PrincipledInputs}(pointer(prog,i)+$offset))
        Hikari.PrincipledInputs($(fields...))
    end
end

function dummy_context()
    n = Hikari.Vec3f(0,0,1)
    tfc = Hikari.TextureFilterContext(Hikari.Point2f(0), Hikari.Point3f(0), n)
    return Hikari.ShaderContext(tfc,n,Hikari.Vec3f(0),Hikari.Vec3f(0),Hikari.Vec3f(1,0,0),n,1f0,Hikari.Vec3f(0),Hikari.Point2f(0))
end

function decode_program(program)
    instructions = copy(program)
    bytes = reinterpret(UInt8, instructions)
    stride = sizeof(eltype(instructions))
    payload = fieldoffset(eltype(instructions),7) # k; header fields remain intact
    i = Int32(1)
    while program[i].op != Hikari.NodeOp.RESULT
        program[i].op == Hikari.NodeOp.PRINCIPLED || error("This control only packs Principled nodes")
        chunks = Hikari.instruction_count(Hikari.PrincipledBSDF)
        after = Hikari.next_node(program,i,chunks)
        for ins in program[i:after-1], arg in ins.arg
            Hikari.reads_stack(arg) && error("This control requires constant shader inputs")
        end
        code(j) = program[i+(j-1)÷Hikari.NODE_ARGS].arg[(j-1)%Hikari.NODE_ARGS+1]
        flags = UInt8((code(7)==Hikari.SHADING_NORMAL_ARG) |
            ((code(26)==Hikari.SHADING_NORMAL_ARG)<<1) |
            ((code(18)==Hikari.SHADING_TANGENT_ARG)<<2))
        p = Hikari.principled_inputs(Float32[], program, i, dummy_context())
        sizeof(p) <= (after-i)*stride-payload || error("Not enough payload bytes in the original node")
        ins = program[i]
        out = (UInt8(0xfd),flags,ins.out[3:end]...)
        instructions[i] = typeof(ins)(ins.op,ins.mode,out,ins.outwide,ins.arg,ins.wide,ins.k)
        start = (Int(i)-1)*stride+payload+1
        copyto!(bytes,start,reinterpret(UInt8,[p]),1,sizeof(p))
        i = after
    end
    return instructions
end

function install_reader!()
    offset = fieldoffset(Hikari.NodeInstr{Hikari.NodeFeature.CLOSURES},7)
    fields = map(fieldnames(Hikari.PrincipledInputs)) do name
        name === :n && return :((flags & 0x01) != 0 ? sc.tfc.n : p.n)
        name === :coat_n && return :((flags & 0x02) != 0 ? sc.tfc.n : p.coat_n)
        name === :tangent && return :((flags & 0x04) != 0 ? sc.dpdus : p.tangent)
        return :(getfield(p,$(QuoteNode(name))))
    end
    Core.eval(Hikari, quote
        @inline function principled_inputs(s,prog,i::Int32,sc::ShaderContext)
            @inbounds ins = prog[i]
            ins.out[1] == 0xfd || error("Decoded-input control reached an unpacked node")
            flags = ins.out[2]
            p = unsafe_load(Ptr{PrincipledInputs}(pointer(prog,i)+$offset))
            return PrincipledInputs($(fields...))
        end
    end)
end

rename_general(x::Symbol) = x === :surface_program_bsdf ? :composition_decoded_general_bsdf : x
rename_general(x::Expr) = Expr(x.head, map(rename_general, x.args)...)
rename_general(x) = x

function run(output;spec,repeats=11,array_storage=false,capacity=16,opaque=false,simple_kinds=false,
             names=("mix_matte_metal","mix_layered_glass"))
    original = Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.principled_inputs)))
    metadata_method = which(Hikari.surface_program_metadata,Tuple{Any,Any,Any,Val{UInt32(0)},Int32})
    original_metadata = Main.Revise.CodeTracking.definition(Expr,metadata_method)
    original isa Expr || error("Cannot restore the input decoder")
    original_metadata isa Expr || error("Cannot restore program metadata")
    original_setup = Main.Revise.CodeTracking.definition(Expr, which(Hikari.surface_setup_slots,
        Tuple{Val{16},Any,Any,Any,Any,Any,Any,Any,Any,Val{true},Val{UInt32(0)}}))
    original_bsdf = Main.Revise.CodeTracking.definition(Expr, which(Hikari.surface_program_bsdf,
        Tuple{Type{Hikari.SurfaceProgramNodes{16,true,UInt32(0),false}},Any,Any,Any,Any,Any,Any,Int32}))
    original_setup isa Expr && original_bsdf isa Expr || error("Cannot restore storage")
    original_two = Main.Revise.CodeTracking.definition(Expr, which(Hikari.surface_program_bsdf,
        Tuple{Type{Hikari.SurfaceProgramNodes{2,true,UInt32(0),false}},Any,Any,Any,Any,Any,Any,Int32}))
    original_two isa Expr || error("Cannot restore class two")
    rows = Any[]
    for settings in filter(row -> haskey(row,:mix) && row.name in names,spec.cases)
        canonical = only(filter(row -> row.name == settings.name,
            MaterialComparison.configuration(joinpath(@__DIR__,"mixed_graph_cases.json")).cases))
        JSON3.write(settings) == JSON3.write(canonical) || error("Bound control requires its exact fixed inputs")
        capacity >= (settings.name == "mix_matte_metal" ? 2 : 6) || error("Insufficient storage")
        opaque && settings.name != "mix_matte_metal" && error("Only matte/metal proves opacity")
        simple_kinds && !opaque && error("Kind pruning requires the opaque matte/metal proof")
        capacity in (2,16) || error("This control supports capacities two and sixteen")
        fixtures = Any[]
        class_method = nothing
        general_methods = Method[]
        try
            baseline = MaterialComparison.fixture(MaterialComparison.principled_material(settings,spec),spec)
            push!(fixtures,baseline)
            Base.invokelatest(MaterialComparison.draw,baseline)
            expected = MaterialComparison.pixels(baseline)
            candidate = MaterialComparison.fixture(MaterialComparison.principled_material(settings,spec),spec)
            push!(fixtures,candidate)
            programs = filter(a -> eltype(a) === Hikari.NodeInstr{Hikari.NodeFeature.CLOSURES},
                candidate.scene.materials.texture_gpu_arrays)
            @test !isempty(programs)
            for program in programs
                copyto!(program,decode_program(Array(program)))
            end
            Mantle.waitidle(candidate.dev)
            general_source = array_storage ? Main.CompositionArrayProbe.install_storage!(original_setup) : original_bsdf
            if capacity != 16 || opaque
                proof = opaque ? Hikari.SURFACE_OPAQUE_STORAGE : UInt32(0)
                clear = simple_kinds ? foldl(|,(UInt32(1)<<k for k in (2,3,5,6,10));init=UInt32(0)) : UInt32(0)
                @eval Hikari function bsdf_kernel_nodes(::Type{M}, api::Mantle.Backend) where {M <: Tuple}
                    original = surface_program_class(M, api)
                    _, looped, mask, emissive = original.parameters
                    SurfaceProgramNodes{$capacity,looped,(mask | $proof) & ~$clear,emissive}
                end
                class_method = which(Hikari.bsdf_kernel_nodes,Tuple{Type{Tuple{}},Mantle.VulkanAPI})
            end
            if capacity == 2
                Core.eval(Hikari,rename_general(general_source))
                helper = Base.invokelatest(getglobal,Hikari,:composition_decoded_general_bsdf)
                append!(general_methods,methods(helper))
                @eval Hikari @inline function surface_program_bsdf(
                        T::Type{SurfaceProgramNodes{2,L,M,E}},s,prog,sc,lut,table,lambda,
                        start::Int32=Int32(1)) where {L,M,E}
                    composition_decoded_general_bsdf(T,s,prog,sc,lut,table,lambda,start)
                end
            end
            install_reader!()
            @eval Hikari @inline function surface_program_metadata(s,prog,sc,::Val{Mask},start::Int32=Int32(1)) where {Mask}
                transparency_sum = Vec3f(0); dispersive = false; i = start
                @inbounds while prog[i].op != NodeOp.RESULT
                    prog[i].op == NodeOp.PRINCIPLED || error("Decoded metadata control requires Principled nodes")
                    p = principled_inputs(s,prog,i,sc)
                    transparency_sum += Vec3f(p.weight*(1f0-clamp(p.alpha,0f0,1f0)))
                    if !iszero(Mask & (UInt32(1)<<7))
                        abbe = max(p.abbe,0f0)
                        dispersion = clamp(p.dispersion,0f0,1f0)
                        thin = p.thin_wall>0.5f0
                        dispersive |= !thin && abbe>0f0 && dispersion>0f0
                    end
                    i = next_node(prog,i,instruction_count(PrincipledBSDF))
                end
                return transparency_sum,dispersive
            end
            first_seconds = @elapsed Base.invokelatest(MaterialComparison.draw,candidate)
            actual = MaterialComparison.pixels(candidate)
            if !isapprox(actual,expected;rtol=5f-5,atol=2f-5)
                mkpath(dirname(output))
                write(output*".actual.f32",reinterpret(UInt8,actual))
                write(output*".expected.f32",reinterpret(UInt8,expected))
                println((max_pixel_error=maximum(abs.(actual.-expected)),mean_ratio=mean(actual)/mean(expected)))
            end
            @test actual ≈ expected rtol=5f-5 atol=2f-5
            for _ in 1:3,f in fixtures
                Base.invokelatest(MaterialComparison.draw,f)
            end
            @test MaterialComparison.pixels(baseline) ≈ expected rtol=5f-5 atol=2f-5
            times = (Float64[],Float64[])
            for i in 1:repeats,j in (isodd(i) ? (1,2) : (2,1))
                push!(times[j],@elapsed Base.invokelatest(MaterialComparison.draw,fixtures[j]))
            end
            @test MaterialComparison.pixels(candidate) ≈ expected rtol=5f-5 atol=2f-5
            @test MaterialComparison.pixels(baseline) ≈ expected rtol=5f-5 atol=2f-5
            record = (;material=settings.name,array_storage,capacity,opaque,simple_kinds,first_seconds,baseline_seconds=median(times[1]),
                candidate_seconds=median(times[2]),max_pixel_error=maximum(abs.(actual.-expected)),frame_seconds=times)
            push!(rows,record)
            mkpath(dirname(output))
            open(io->JSON3.write(io,rows),output,"w")
            println((;record.material,record.baseline_seconds,record.candidate_seconds,record.max_pixel_error))
            flush(stdout)
        finally
            class_method === nothing || Base.delete_method(class_method)
            capacity == 2 && Core.eval(Hikari,original_two)
            foreach(Base.delete_method,general_methods)
            Core.eval(Hikari,original)
            Core.eval(Hikari,original_metadata)
            if array_storage
                Core.eval(Hikari,original_setup)
                Core.eval(Hikari,original_bsdf)
            end
            foreach(MaterialComparison.release,fixtures)
        end
    end
    return nothing
end
end
