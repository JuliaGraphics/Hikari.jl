module CompositionArrayProbe
using Hikari, Mantle, JSON3, Statistics, Test
using ..MaterialComparison

struct ArraySpan{A <: AbstractVector} <: Hikari.AbstractLobeStorage
    data::A
    count::Int32
end
Base.getindex(s::ArraySpan, i) = s.data[i]
Base.length(s::ArraySpan) = Int(s.count)
Base.map(f, s::ArraySpan) = ArraySpan(map(f, s.data), s.count)
@inline function Hikari.share(s::ArraySpan, restriction::Hikari.Scattering)
    total = 0f0
    @inbounds for i in 1:s.count
        total += Hikari.share(s.data[i], restriction)
    end
    return total
end
@inline Hikari.mixture(s::ArraySpan, wo, wi, ns, lambda, restriction::Hikari.Scattering = Hikari.AnyScattering()) =
    Hikari.surface_mixture(s.data, s.count, wo, wi, ns, lambda, restriction)
@inline Hikari.pick_sample(s::ArraySpan, target, before, wo, u, lambda, restriction::Hikari.Scattering = Hikari.AnyScattering()) =
    Hikari.surface_pick_sample(s.data, s.count, target, before, wo, u, lambda, restriction)
Hikari.terminates_at_vertex(::Hikari.LobeBSDF{<:ArraySpan}) = false
Hikari.subsurface_slot(::Type{<:ArraySpan}) = 0

struct FlatStorage{W,A <: AbstractVector{Float32}} <: AbstractVector{W}
    data::A
end
Base.size(s::FlatStorage) = (length(s.data) ÷ 27,)
@inline function flat_storage(::Val{N}, ::Val{L}, ::Val{M}) where {N,L,M}
    data = Hikari.MVector{27*N,Float32}(undef)
    @inbounds for i in 1:(27*N)
        data[i] = 0f0
    end
    return FlatStorage{Hikari.SurfaceStoredWeight{L,M},typeof(data)}(data)
end
@inline function Base.setindex!(s::FlatStorage{W}, w::W, i) where {W}
    at = (i - Int32(1)) * Int32(27)
    @inbounds begin
        s.data[at+1] = reinterpret(Float32, w.lobe.kind)
        for j in 1:15
            s.data[at+1+j] = w.lobe.data[j]
        end
        for j in 1:4
            s.data[at+16+j] = w.weight.data[j]
        end
        s.data[at+21] = w.sample_weight
        for j in 1:3
            s.data[at+21+j] = w.n[j]
            s.data[at+24+j] = w.t[j]
        end
    end
    return w
end
@inline function Base.getindex(s::FlatStorage{Hikari.SurfaceStoredWeight{L,M}}, i) where {L,M}
    at = (i - Int32(1)) * Int32(27)
    @inbounds return Hikari.Weighted(
        Hikari.SurfaceStoredLobe{L,M}(reinterpret(UInt32,s.data[at+1]), ntuple(j -> s.data[at+1+j], Val(15))),
        Hikari.SpectralRadiance(ntuple(j -> s.data[at+16+j], Val(4))), s.data[at+21],
        Hikari.Vec3f(ntuple(j -> s.data[at+21+j], Val(3))),
        Hikari.Vec3f(ntuple(j -> s.data[at+24+j], Val(3))))
end

function array_setup(ex, flat, uninitialized = false)
    ex isa Expr || return ex
    ex.head === :call && ex.args == [:Tuple, :storage] && return :storage
    if uninitialized && ex.head === :call && ex.args[1] isa Expr &&
            ex.args[1].head === :curly && first(ex.args[1].args) === :MVector
        return :(MVector{N,SurfaceStoredWeight{Looped,Mask}}(undef))
    end
    if flat && ex.head === :call && ex.args[1] isa Expr &&
            ex.args[1].head === :curly && first(ex.args[1].args) === :MVector
        return :($(flat_storage)(Val(N), mode, mask))
    end
    return Expr(ex.head, map(x -> array_setup(x, flat, uninitialized), ex.args)...)
end

function install_storage!(original_setup; flat = false, uninitialized = false)
    Core.eval(Hikari, array_setup(original_setup, flat, uninitialized))
    definition = quote
    @inline function surface_program_bsdf(
            ::Type{SurfaceProgramNodes{N,L,M,E}}, s, prog, sc, lut, table, lambda,
            start::Int32=Int32(1)) where {N,L,M,E}
        transparency, dispersive = surface_program_metadata(s, prog, sc, Val(M), start)
        alpha = 1f0 - _avg(transparency)
        scale = alpha > 1f-6 ? 1f0 / alpha : 0f0
        slots, emission, count = surface_setup_slots(Val(N), s, prog, start, sc, lut, table,
            lambda, scale, Val(L), Val(M))
        total = if $uninitialized
            weight = 0f0
            @inbounds for j in 1:count
                weight += slots[j].sample_weight
            end
            weight
        else
            reduce(+, ntuple(j -> slots[j].sample_weight, Val(N)); init=0f0)
        end
        light = E && _max(emission) > 0f0 ?
            uplift_rgb_illuminant(table, RGBSpectrum(emission...), lambda) / D65_PHOTOMETRIC : SpectralRadiance(0f0)
        return LobeBSDF($(ArraySpan)(slots, count), total, lambda, light, dispersive)
    end
    end
    Core.eval(Hikari, definition)
    return definition
end

function tuple_reference(ex, hits)
    ex isa Expr || return ex
    if ex.head === :if && length(ex.args) == 3 &&
            ex.args[2] == :(Tuple(storage)) && ex.args[3] === :storage
        hits[] += 1
        return :(Tuple(storage))
    end
    return Expr(ex.head, map(x -> tuple_reference(x, hits), ex.args)...)
end

function run(output; spec, repeats = 7, capacity = 16, flat = false,
             storage = true, nofilm = false, profile = false, uninitialized = false,
             reference_tuple = false, names = ("mix_matte_metal", "mix_layered_glass"))
    uninitialized && (flat || !storage) && error("Uninitialized storage control requires structured local arrays")
    reference_tuple && storage && error("Compare the previous tuple backing against production storage")
    capacity in (2,5) && error("Those classes reserve a single-node layout; use a general capacity for this control")
    setup_method = which(Hikari.surface_setup_slots,
        Tuple{Val{16},Any,Any,Any,Any,Any,Any,Any,Any,Val{true},Val{UInt32(0)}})
    original_setup = Main.Revise.CodeTracking.definition(Expr, setup_method)
    bsdf_method = which(Hikari.surface_program_bsdf,
        Tuple{Type{Hikari.SurfaceProgramNodes{16,true,UInt32(0),false}},Any,Any,Any,Any,Any,Any,Int32})
    original_bsdf = Main.Revise.CodeTracking.definition(Expr, bsdf_method)
    original_setup isa Expr && original_bsdf isa Expr || error("Cannot restore shared storage")
    hits = Ref(0)
    previous_setup = reference_tuple ? tuple_reference(original_setup, hits) : original_setup
    reference_tuple && hits[] != 1 && error("Expected one production storage selection")
    film_method = which(Hikari.has_film, Tuple{Hikari.F82Tint{Hikari.SpectralRadiance}})
    original_film = Main.Revise.CodeTracking.definition(Expr, film_method)
    original_film isa Expr || error("Cannot restore thin-film dispatch")
    resources(f) = [(; bytes = length(d.launch.compiled.spirv_bytes),
        native = Mantle.pipeline_exec_stats(f.dev, d.launch.pipeline))
        for plan in Hikari.allplans(f.vp.state.plans) for pass in plan.passes
        if pass.pass.name == "shade" for d in pass.dispatches]
    rows = Any[]
    for settings in filter(row -> row.name in names, spec.cases)
        canonical = only(filter(row -> row.name == settings.name,
            MaterialComparison.configuration(joinpath(@__DIR__, "mixed_graph_cases.json")).cases))
        JSON3.write(settings) == JSON3.write(canonical) || error("This storage control requires its exact fixed inputs")
        capacity >= (settings.name == "mix_matte_metal" ? 2 : 6) || error("Insufficient storage")
        fixtures = Any[]
        class_method = nothing
        queue_method = nothing
        try
            reference_tuple && Core.eval(Hikari, previous_setup)
            baseline = MaterialComparison.fixture(MaterialComparison.principled_material(settings, spec), spec)
            push!(fixtures, baseline)
            Base.invokelatest(MaterialComparison.draw, baseline)
            expected = MaterialComparison.pixels(baseline)
            reference_tuple && Core.eval(Hikari, original_setup)
            if nofilm
                @eval Hikari @inline has_film(f::Union{F82Tint, SchlickFresnel, PhysicalConductorFresnel}) = false
            end
            if storage
                install_storage!(original_setup; flat, uninitialized)
            end
            @eval Hikari function bsdf_kernel_nodes(::Type{M}, api::Mantle.Backend) where {M <: Tuple}
                original = surface_program_class(M, api)
                _, looped, mask, emissive = original.parameters
                return SurfaceProgramNodes{$capacity, looped, mask, emissive}
            end
            class_method = which(Hikari.bsdf_kernel_nodes, Tuple{Type{Tuple{}},Mantle.VulkanAPI})
            if capacity == 8
                @eval Hikari shade_queues(q::ClosureQueue{SurfaceProgramNodes{8,L,K,E}}) where {L,K,E} = (q,)
                queue_method = which(Hikari.shade_queues,
                    Tuple{Hikari.ClosureQueue{Hikari.SurfaceProgramNodes{8,true,UInt32(0),false}}})
            end
            candidate = Base.invokelatest(MaterialComparison.fixture,
                MaterialComparison.principled_material(settings, spec), spec)
            push!(fixtures, candidate)
            first_seconds = @elapsed Base.invokelatest(MaterialComparison.draw, candidate)
            actual = MaterialComparison.pixels(candidate)
            @test actual ≈ expected rtol=5f-5 atol=2f-5
            for _ in 1:3, f in fixtures
                Base.invokelatest(MaterialComparison.draw, f)
            end
            times = (Float64[], Float64[])
            for i in 1:repeats, j in (isodd(i) ? (1,2) : (2,1))
                push!(times[j], @elapsed Base.invokelatest(MaterialComparison.draw, fixtures[j]))
            end
            @test MaterialComparison.pixels(candidate) ≈ expected rtol=5f-5 atol=2f-5
            record = (; material=settings.name, capacity, flat, storage, nofilm, uninitialized, reference_tuple, first_seconds,
                baseline_seconds=median(times[1]), candidate_seconds=median(times[2]),
                max_pixel_error=maximum(abs.(actual.-expected)), frame_seconds=times,
                resources = profile ? map(resources, fixtures) : nothing)
            push!(rows, record)
            mkpath(dirname(output))
            open(io -> JSON3.write(io,rows),output,"w")
            println((; record.material, record.baseline_seconds, record.candidate_seconds, record.max_pixel_error))
            flush(stdout)
        finally
            class_method === nothing || Base.delete_method(class_method)
            queue_method === nothing || Base.delete_method(queue_method)
            Core.eval(Hikari, original_setup)
            Core.eval(Hikari, original_bsdf)
            nofilm && Core.eval(Hikari, original_film)
            foreach(MaterialComparison.release, fixtures)
        end
    end
    return nothing
end
end
