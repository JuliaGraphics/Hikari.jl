module UniformSlotsProbe
using Hikari, Mantle, JSON3

@inline function setslot!(data, w, j)
    b = 27 * (j-1)
    values = (w.lobe.data..., w.weight.data..., w.sample_weight, w.n..., w.t...)
    @inbounds data[b+1] = w.lobe.kind
    @inbounds for i in 1:26
        data[b+1+i] = reinterpret(UInt32, values[i])
    end
    data
end
@inline function getslot(data, j, ::Val{L}, ::Val{M}) where {L,M}
    b = 27 * (j-1)
    f(k) = @inbounds reinterpret(Float32, data[b+k])
    lobe = Hikari.SurfaceStoredLobe{L,M}(@inbounds(data[b+1]), ntuple(k -> f(k+1), Val(15)))
    Hikari.Weighted(lobe, Hikari.SpectralRadiance(ntuple(k -> f(k+16), Val(4))),
        f(21), Hikari.Vec3f(f(22),f(23),f(24)), Hikari.Vec3f(f(25),f(26),f(27)))
end
@generated getslots(data, ::Val{N}, mode, mask) where {N} =
    Expr(:tuple, (:(getslot(data, $i, mode, mask)) for i in 1:N)...)

function rewrite(x)
    x isa Expr || return x
    x.head == :(=) && x.args[1] == :storage && return :(
        storage = MVector{27*N,UInt32}(ntuple(_ -> UInt32(0), Val(27*N))))
    x == :(Tuple(storage)) && return :($getslots(storage, Val(N), mode, mask))
    Expr(x.head, (rewrite(a) for a in x.args)...)
end

function run(output)
    original = Main.Revise.CodeTracking.definition(Expr,
        first(filter(m -> occursin("Looped", string(m.sig)), methods(Hikari.surface_setup_slots))))
    original isa Expr || error("Cannot recover storage setup")
    cases = [p for batch in Main.cycles_batches() for p in batch
             if p[1] in ("subsurface_node_zero_channel", "mix_subsurface_diffuse")]
    before = Set(methods(Hikari.surface_append!))
    try
        Core.eval(Hikari, quote
            @inline surface_append!(storage::MVector{B,UInt32}, ::Tuple{}, at, table, lambda, scale, mode, mask) where {B} = at
            @inline function surface_append!(storage::MVector{B,UInt32}, terms::Tuple, at, table, lambda, scale, mode, mask) where {B}
                term = first(terms)
                if term.sample_weight > 0f0
                    $setslot!(storage, surface_store(lift(term, scale, table, lambda), mode, mask), at)
                    at += Int32(1)
                end
                surface_append!(storage, Base.tail(terms), at, table, lambda, scale, mode, mask)
            end
        end)
        Core.eval(Hikari, rewrite(original))
        actual = Base.invokelatest(Main.render_cycles_batch, cases; samples = 64)
        records = [(; name, mean = Main.cycles_mean(actual, Main.cycles_sphere_pixels(k)))
                   for (k, (name, _)) in enumerate(cases)]
        open(io -> JSON3.write(io, records), output, "w")
        println("UNIFORM_SLOTS ", records); flush(stdout)
    finally
        Core.eval(Hikari, original)
        for method in setdiff(Set(methods(Hikari.surface_append!)), before)
            Base.delete_method(method)
        end
    end
    nothing
end
end
