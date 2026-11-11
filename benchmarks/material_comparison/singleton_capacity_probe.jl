module SingletonCapacityProbe
using Hikari, Mantle, KernelAbstractions, JSON3, Statistics, Test
using ..MaterialComparison
const Probe=MaterialComparison

@kernel inbounds=true function counts!(sizes,q,programs)
    if @index(Global)==1
        program=q.inputs.program[q.program_offset+Int32(1)]
        hint=Hikari.closure_capacity_hint(programs[program][Int32(1)])
        n=q.sizes[q.slot]
        sizes[1]=hint==1 ? n : Int32(0)
        sizes[2]=hint==2 ? n : Int32(0)
        sizes[3]=hint==5 ? n : Int32(0)
        sizes[4]=hint==0 ? n : Int32(0)
    end
end

function uniform!(g,state,refs,q)
    p=state.shader_partition
    Mantle.dispatch!(g,counts!,(p.sizes,q,Hikari.closure_programs(refs.materials)),1;group=1,name="capacity-counts")
    choices=Hikari.shade_queues(q)
    map(enumerate(choices) |> Tuple) do (bin,choice)
        original=choice.queue
        base=typeof(original)(original.items,p.sizes,Int32(bin),original.offset,original.inputs,original.program_offset)
        typeof(choice)(base)
    end
end

function with_singletons(f)
    original=Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.partitioned_shade_queues!)))
    original isa Expr || error("Cannot recover partitioned shading graph")
    world=Base.get_world_counter()
    try
        Core.eval(Hikari,quote
            function partitioned_shade_queues!(g,state::VolPathState,@nospecialize(refs),q)
                choices=shade_queues(q)
                length(choices)==1 && return choices
                length(refs.materials.data[q.slot])==1 && return $uniform!(g,state,refs,q)
                Base.invoke_in_world($world,partitioned_shade_queues!,g,state,refs,q)
            end
        end)
        Base.invokelatest(f)
    finally
        Core.eval(Hikari,original)
    end
end

function run(output;device,repeats=31)
    isdefined(Hikari,:vp_shader_capacity_counts_kernel!) && error(
        "This historical control requires source before single-material capacity counts were integrated.")
    Mantle.defaultdevice!(device)
    spec=Probe.configuration()
    results=Any[]
    for row in spec.cases
        baseline=Probe.fixture(Probe.principled_material(row),spec)
        candidate=nothing
        try
            Base.invokelatest(Probe.draw,baseline)
            expected=Probe.pixels(baseline)
            with_singletons() do
                candidate=Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec)
                first_seconds=@elapsed Base.invokelatest(Probe.draw,candidate)
                actual=Probe.pixels(candidate)
                @test all(isfinite,actual)
                @test actual==expected
                @test Probe.shade_signatures(candidate)==Probe.shade_signatures(baseline)
                times=(Float64[],Float64[])
                for _ in 1:5,f in (baseline,candidate)
                    Base.invokelatest(Probe.draw,f)
                end
                for i in 1:repeats,k in (isodd(i) ? (1,2) : (2,1))
                    push!(times[k],@elapsed Base.invokelatest(Probe.draw,k==1 ? baseline : candidate))
                end
                @test Probe.pixels(candidate)==actual
                result=(;name=row.name,first_seconds,frame_seconds=times,median_seconds=map(median,times),
                         max_pixel_error=maximum(abs.(actual.-expected)))
                push!(results,result)
                mkpath(dirname(output));open(io->JSON3.write(io,results),output,"w")
                println(row.name," grouped/single-material capacity ms: ",result.median_seconds.*1000)
                flush(stdout)
            end
        finally
            candidate===nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    nothing
end
end
