module CapacityPartitionProbe
using Hikari, Mantle, Raycore, KernelAbstractions, Atomix, JSON3, Statistics, Test
using ..MaterialComparison
const Probe = MaterialComparison
const KI = KernelAbstractions.KernelInterface

# Control: classify hits once, rather than launch all four capacity kernels
# over every hit and reject three classes inside those kernels.
@kernel inbounds=true function route!(q,hits,accel,interfaces,materials,items,sizes,capacity::Int32)
    j = @index(Global)
    if j <= Hikari.queuelength(q)
        ref = q[j]
        sh = hits.items[ref.idx]
        primitive = Raycore.hit_primitive(accel,sh.hit)
        key = interfaces[Hikari.resolve_mi_idx(sh.hit.custom,primitive)].material
        program = q.inputs.program[q.program_offset+key.vec_idx]
        hint = Hikari.closure_capacity_hint(Hikari.closure_programs(materials)[program][Int32(1)])
        bin = hint==1 ? Int32(1) : hint==2 ? Int32(2) : hint==5 ? Int32(3) : Int32(4)
        pos = Atomix.@atomic :monotonic sizes[bin] += Int32(1)
        pos <= capacity && (items[(bin-Int32(1))*capacity+pos]=ref.idx)
    end
end

@kernel inbounds=true function route_grouped!(q,hits,accel,interfaces,materials,items,sizes,capacity::Int32)
    j = @index(Global)
    active = j <= Hikari.queuelength(q)
    idx = Int32(0); target = Int32(0)
    if active
        ref = q[j]
        idx = ref.idx
        sh = hits.items[idx]
        primitive = Raycore.hit_primitive(accel,sh.hit)
        key = interfaces[Hikari.resolve_mi_idx(sh.hit.custom,primitive)].material
        program = q.inputs.program[q.program_offset+key.vec_idx]
        hint = Hikari.closure_capacity_hint(Hikari.closure_programs(materials)[program][Int32(1)])
        target = hint==1 ? Int32(1) : hint==2 ? Int32(2) : hint==5 ? Int32(3) : Int32(4)
    end
    lane = KI.get_sub_group_local_id(Int32)
    preceding = (UInt64(1) << (lane-Int32(1)))-UInt64(1)
    # Every work-item participates, including the padded lanes. The votes and
    # shuffles are outside divergent branches; a subgroup's first lane reserves
    # one contiguous span for each of the four classes.
    for bin in Int32(1):Int32(4)
        mask = KI.sub_group_ballot(active && target==bin)
        base = Int32(0)
        if lane==Int32(1) && !iszero(mask)
            count = Int32(count_ones(mask))
            base = (Atomix.@atomic :monotonic sizes[bin] += count)-count
        end
        base = KI.shfl(base,Int32(1))
        if active && target==bin
            pos = base+Int32(count_ones(mask & preceding))+Int32(1)
            pos <= capacity && (items[(bin-Int32(1))*capacity+pos]=idx)
        end
    end
end

function routed(q::Hikari.CapacityQueue{M,C},bin,items,sizes,capacity) where {M,C}
    original = q.queue
    # Retain the exact existing queue/ref signature and its compiled kernels.
    # Only the index segment and DeviceRange count change. The cheap capacity
    # guard stays, but no rejected class launches work for these hits.
    base = typeof(original)(items,sizes,Int32(bin),Int32((bin-1)*capacity),original.inputs,original.program_offset)
    return typeof(q)(base)
end

function partition!(g,state,refs,q;kernel=route!)
    choices = Hikari.shade_queues(q)
    length(choices)==1 && return choices
    capacity = state.material_queues.capacity
    items = Hikari.alloc!(state.per_material_memory,Int32,4*Int(capacity))
    sizes = Hikari.alloc!(state.per_material_memory,Int32,4)
    Mantle.dispatch!(g,Hikari.zero_size_counters_kernel!,((sizes,),),1;group=1,name="partition-reset")
    Mantle.dispatch!(g,kernel,
        (q,state.surface_hits,refs.accel,refs.media_interfaces,Hikari.closure_set(refs.materials),items,sizes,capacity),
        Mantle.DeviceRange(q.sizes;max=Int(capacity),at=q.slot);group=128,name="partition")
    return map(enumerate(choices) |> Tuple) do (bin,choice)
        routed(choice,bin,items,sizes,capacity)
    end
end
partition_grouped!(g,state,refs,q) = partition!(g,state,refs,q;kernel=route_grouped!)

range(q::Hikari.ClosureQueue,queues,slot) = Mantle.DeviceRange(q.sizes;max=Int(queues.capacity),at=q.slot)
range(q::Hikari.CapacityQueue,queues,slot) = range(q.queue,queues,slot)
range(q,queues,slot) = Mantle.DeviceRange(queues.sizes;max=Int(queues.capacity),at=slot)

function callname(x)
    x isa Symbol && return x
    x isa GlobalRef && return x.name
    x isa Expr && x.head==:. && return last(x.args).value
    nothing
end
function rewrite(x,inside=false,partition=partition!)
    x isa Expr || return x
    if x.head==:for && first(x.args) isa Expr && first(first(x.args).args)==:shade_queue
        return Expr(:for,rewrite(x.args[1],false,partition),rewrite(x.args[2],true,partition))
    elseif x.head==:call && callname(first(x.args))==:shade_queues
        return :($partition(g,state,refs,$(x.args[2])))
    elseif inside && x.head==:call && callname(first(x.args))==:DeviceRange
        return :($range(shade_queue,queues,slot))
    end
    return Expr(x.head,(rewrite(a,inside,partition) for a in x.args)...)
end

function run(output;device,repeats=15,grouped=false,names=("matte","rough_glass","rough_metal","plastic","layered_principled"))
    isdefined(Hikari,:partitioned_shade_queues!) && error(
        "This historical control requires source before bounded shader queue partitioning was integrated.")
    Mantle.defaultdevice!(device)
    original = Main.Revise.CodeTracking.definition(Expr,only(methods(Hikari.shade_pass!)))
    original isa Expr || error("Cannot recover shade_pass! for restoration")
    replacement = rewrite(original,false,grouped ? partition_grouped! : partition!)
    spec = Probe.configuration()
    records = Any[]
    for row in spec.cases
        row.name in names || continue
        baseline = Probe.fixture(Probe.principled_material(row),spec)
        candidate = nothing
        try
            Base.invokelatest(Probe.draw,baseline)
            expected = Probe.pixels(baseline)
            Core.eval(Hikari,replacement)
            candidate = Base.invokelatest(Probe.fixture,Probe.principled_material(row),spec)
            before = Mantle.Lava.compile_stats()
            first_seconds = @elapsed Base.invokelatest(Probe.draw,candidate)
            after = Mantle.Lava.compile_stats()
            actual = Probe.pixels(candidate)
            @test all(isfinite,actual)
            @test actual≈expected rtol=5f-5 atol=2f-5
            @test Probe.shade_signatures(candidate)==Probe.shade_signatures(baseline)
            @test any(pp->pp.pass.name=="partition",candidate.vp.state.plans.sample.passes)
            times = (Float64[],Float64[])
            for _ in 1:5,f in (baseline,candidate)
                Base.invokelatest(Probe.draw,f)
            end
            for i in 1:repeats,k in (isodd(i) ? (1,2) : (2,1))
                push!(times[k],@elapsed Base.invokelatest(Probe.draw,k==1 ? baseline : candidate))
            end
            @test Probe.pixels(candidate)≈actual rtol=5f-5 atol=2f-5
            extra_bytes = sum(b->length(Mantle.storage(b))*sizeof(eltype(Mantle.storage(b))),candidate.vp.state.per_material_memory.owned;init=0)-
                          sum(b->length(Mantle.storage(b))*sizeof(eltype(Mantle.storage(b))),baseline.vp.state.per_material_memory.owned;init=0)
            result = (;name=row.name,first_seconds,spirv_misses=after.misses-before.misses,
                       frame_seconds=times,median_seconds=map(median,times),
                       extra_bytes,max_pixel_error=maximum(abs.(actual.-expected)))
            push!(records,result)
            mkpath(dirname(output));open(io->JSON3.write(io,records),output,"w")
            println(row.name," reject/partition ms: ",result.median_seconds.*1000,
                    "; image error ",result.max_pixel_error,"; extra buffers ",extra_bytes)
            flush(stdout)
        finally
            Core.eval(Hikari,original)
            candidate===nothing || Probe.release(candidate)
            Probe.release(baseline)
        end
    end
    nothing
end
end
