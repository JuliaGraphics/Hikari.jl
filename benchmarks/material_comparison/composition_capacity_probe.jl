module CompositionCapacityProbe
using Hikari, Mantle, JSON3, Statistics, Test
using ..MaterialComparison

rename_general(x::Symbol) = x === :surface_program_bsdf ? :composition_probe_general_bsdf : x
rename_general(x::Expr) = Expr(x.head, map(rename_general, x.args)...)
rename_general(x) = x

@inline has_thin(::Val{M}) where {M} = !iszero(M & (UInt32(1)<<10))
function prune_thin(ex, count)
    ex isa Expr || return ex
    if ex == :(p.thin_wall > 0.5f0)
        count[] += 1
        return :($(has_thin)(mask) && $ex)
    end
    return Expr(ex.head,map(x -> prune_thin(x,count),ex.args)...)
end

function rolled_loops(x)
    x isa Expr || return x
    args = map(rolled_loops, x.args)
    if x.head in (:for, :while)
        args[2] = Expr(:block, args[2], Expr(:loopinfo, (Symbol("llvm.loop.unroll.disable"),)))
    end
    return Expr(x.head, args...)
end

# These are controlled constant-input graphs, not a general capacity classifier.
# Matte + metal needs two live records; layered + glass needs at most six.
# The probe changes only queue storage, retaining the program and BSDF equations.
function resources(f, directory = nothing)
    launches = unique([d.launch for plan in Hikari.allplans(f.vp.state.plans)
        for pass in plan.passes if pass.pass.name == "shade" for d in pass.dispatches])
    if directory !== nothing
        mkpath(directory)
        for (i, launch) in enumerate(launches)
            write(joinpath(directory, "shade_$i.spv"), launch.compiled.spirv_bytes)
        end
    end
    return [(; bytes = length(l.compiled.spirv_bytes),
               native = Mantle.pipeline_exec_stats(f.dev, l.pipeline)) for l in launches]
end

function run(output; spec, repeats = 31, warmups = 5, profile = false, partition = true,
             unroll = false, rolled = false, dump = false,
             clear = UInt32(0),
             experiments = (("mix_matte_metal", 4, UInt32(0)), ("mix_layered_glass", 8, UInt32(0))))
    rows = Any[]
    for (name, capacity, proof) in experiments
        settings = only(filter(row -> row.name == name, spec.cases))
        canonical = only(filter(row -> row.name == name,
            MaterialComparison.configuration(joinpath(@__DIR__, "mixed_graph_cases.json")).cases))
        JSON3.write(settings) == JSON3.write(canonical) || error("This storage control requires its exact fixed inputs")
        capacity >= (name == "mix_matte_metal" ? 2 : 6) || error("Insufficient storage for this control")
        iszero(proof) || name == "mix_matte_metal" || error("Only the matte/metal control proves opacity")
        clear in (UInt32(0),UInt32(1)<<10) || error("Only thin-wall exclusion is proven by these fixed inputs")
        fixtures = Any[]
        method = nothing
        queue_method = nothing
        restore_two = nothing
        general_methods = Method[]
        traversal_methods = Method[]
        rolled_sources = Expr[]
        original_layers = nothing
        try
            baseline = MaterialComparison.fixture(MaterialComparison.principled_material(settings, spec), spec)
            push!(fixtures, baseline)
            MaterialComparison.draw(baseline)
            expected = MaterialComparison.pixels(baseline)
            @eval Hikari function bsdf_kernel_nodes(::Type{M}, api::Mantle.Backend) where {M <: Tuple}
                original = surface_program_class(M, api)
                original === nothing && error("This control supports shared BSDF programs only")
                _, looped, mask, emissive = original.parameters
                return SurfaceProgramNodes{$capacity, looped, (mask | $proof) & ~$clear, emissive}
            end
            method = which(Hikari.bsdf_kernel_nodes, Tuple{Type{Tuple{}}, Mantle.VulkanAPI})
            if !iszero(clear)
                layers = which(Hikari.surface_principled_layers!,Tuple{Vararg{Any,12}})
                original_layers = Main.Revise.CodeTracking.definition(Expr,layers)
                original_layers isa Expr || error("Cannot restore layer setup")
                count = Ref(0)
                replacement = prune_thin(original_layers,count)
                count[] == 1 || error("Thin-wall guard was not found exactly once")
                Core.eval(Hikari,replacement)
            end
            if rolled
                for (f, signature) in ((Hikari.surface_mixture, Tuple{Vararg{Any,7}}),
                        (Hikari.surface_pick_sample, Tuple{Vararg{Any,8}}),
                        (Hikari.share, Tuple{Hikari.SurfaceLobeSpan,Hikari.Scattering}))
                    source = Main.Revise.CodeTracking.definition(Expr, which(f, signature))
                    source isa Expr || error("Cannot restore rolled traversal")
                    push!(rolled_sources, source)
                    Core.eval(Hikari, rolled_loops(source))
                end
            end
            if !partition && capacity == 8
                @eval Hikari shade_queues(q::ClosureQueue{SurfaceProgramNodes{8,L,K,E}}) where {L,K,E} = (q,)
                queue_method = which(Hikari.shade_queues,
                    Tuple{Hikari.ClosureQueue{Hikari.SurfaceProgramNodes{8,true,UInt32(0),false}}})
            end
            if unroll
                before_pick = Set(methods(Hikari.surface_pick_sample))
                before_share = Set(methods(Hikari.share))
                @eval Hikari @inline function surface_pick_sample(lobes::NTuple{$capacity,W}, count,
                        target, before, wo, u, lambda, s) where {W <: SurfaceStoredWeight}
                    Base.Cartesian.@nexprs $capacity j -> begin
                        if j <= count
                            w = lobes[j]; sw = share(w, s); after = before + sw
                            if sw > 0.0f0 && (target < after || j == count)
                                uc = clamp((target - before) / max(sw, 1.0f-12), 0.0f0, 0.99999994f0)
                                n, t, b = lobe_frame(w.lobe, w.n, w.t, wo)
                                wi, f, pdf, flags, eta = lobe_sample(w.lobe, world_to_local(wo, n, t, b), u, uc, lambda, s)
                                return (normalize(local_to_world(wi, n, t, b)), w.weight * f,
                                    pdf, flags, eta, sw, n, terminates_secondary(w.lobe, flags))
                            end
                            before = after
                        end
                    end
                    return pick_sample((), target, before, wo, u, lambda, s)
                end
                @eval Hikari @inline function share(span::SurfaceLobeSpan{NTuple{$capacity,W}}, restriction::Scattering) where {W <: SurfaceStoredWeight}
                    total = 0.0f0
                    Base.Cartesian.@nexprs $capacity j -> begin
                        j <= span.count && (total += share(span.data[j], restriction))
                    end
                    return total
                end
                append!(traversal_methods, setdiff(Set(methods(Hikari.surface_pick_sample)), before_pick))
                append!(traversal_methods, setdiff(Set(methods(Hikari.share)), before_share))
            end
            if capacity == 2
                # Class two normally proves one Principled. Invoke the general
                # tagged representation explicitly for this two-node control.
                two = which(Hikari.surface_program_bsdf,
                    Tuple{Type{Hikari.SurfaceProgramNodes{2,true,UInt32(0),false}},Any,Any,Any,Any,Any,Any,Int32})
                restore_two = Main.Revise.CodeTracking.definition(Expr, two)
                restore_two isa Expr || error("Cannot restore class-two dispatch")
                general = which(Hikari.surface_program_bsdf,
                    Tuple{Type{Hikari.SurfaceProgramNodes{4,true,UInt32(0),false}},Any,Any,Any,Any,Any,Any,Int32})
                source = Main.Revise.CodeTracking.definition(Expr, general)
                source isa Expr || error("Cannot recover the general tagged representation")
                Core.eval(Hikari, rename_general(source))
                helper = Base.invokelatest(getglobal, Hikari, :composition_probe_general_bsdf)
                append!(general_methods, methods(helper))
                @eval Hikari @inline function surface_program_bsdf(
                        T::Type{SurfaceProgramNodes{2,L,M,E}}, s, prog, sc, lut, table, lambda,
                        start::Int32=Int32(1)) where {L,M,E}
                    composition_probe_general_bsdf(T, s, prog, sc, lut, table, lambda, start)
                end
            end
            candidate = Base.invokelatest(MaterialComparison.fixture,
                MaterialComparison.principled_material(settings, spec), spec)
            push!(fixtures, candidate)
            first = @elapsed Base.invokelatest(MaterialComparison.draw, candidate)
            actual = MaterialComparison.pixels(candidate)
            @test actual ≈ expected rtol = 5f-5 atol = 2f-5
            for _ in 1:warmups, f in fixtures
                Base.invokelatest(MaterialComparison.draw, f)
            end
            times = (Float64[], Float64[])
            for i in 1:repeats, j in (isodd(i) ? (1, 2) : (2, 1))
                push!(times[j], @elapsed Base.invokelatest(MaterialComparison.draw, fixtures[j]))
            end
            @test MaterialComparison.pixels(candidate) ≈ expected rtol = 5f-5 atol = 2f-5
            record = (; material = name, capacity, proof, clear, partition, unroll, rolled, first_seconds = first,
                baseline_seconds = median(times[1]), candidate_seconds = median(times[2]),
                ratio = median(times[2]) / median(times[1]),
                max_pixel_error = maximum(abs.(actual .- expected)), frame_seconds = times,
                resources = profile ? [resources(f, dump ? output * ".modules/$(name)_$(capacity)_$(proof)_$side" : nothing)
                    for (side, f) in enumerate(fixtures)] : nothing)
            push!(rows, record)
            mkpath(dirname(output))
            open(io -> JSON3.write(io, rows), output, "w")
            println((; record.material, capacity, proof, record.baseline_seconds,
                       record.candidate_seconds, record.ratio, record.max_pixel_error))
            flush(stdout)
        finally
            method === nothing || Base.delete_method(method)
            queue_method === nothing || Base.delete_method(queue_method)
            restore_two === nothing || Core.eval(Hikari, restore_two)
            foreach(Base.delete_method, general_methods)
            foreach(Base.delete_method, traversal_methods)
            foreach(source -> Core.eval(Hikari, source), rolled_sources)
            original_layers === nothing || Core.eval(Hikari,original_layers)
            foreach(MaterialComparison.release, fixtures)
        end
    end
    return nothing
end
end
