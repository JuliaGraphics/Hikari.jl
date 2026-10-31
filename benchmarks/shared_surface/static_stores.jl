# Optional measured variant: include into Hikari after load.jl.
# Constant block stores preserved image agreement but cost about 21% in the
# layered Principled 512x512, 8 spp comparison on RTX 4000 Ada.
@inline surface_append!(storage, ::Tuple{}, at, table, lambda, scale, mode, mask) = at
@inline function surface_append!(storage, terms::Tuple, at, table, lambda, scale, mode, mask)
    packed = map(w -> surface_store(lift(w, scale, table, lambda), mode, mask), terms)
    return surface_write_packed!(storage, packed, at)
end

# Each node produces a fixed-size block. Choose its starting slot once and use
# constant stores within the block, so SROA can promote the local array instead
# of lowering every field write as a dynamically indexed register selection.
@generated function surface_write_packed!(storage::MVector{N, W}, terms::NTuple{K, W}, at::Int32) where {N, W, K}
    (K == 0 || K > N) && return :(at)
    function stores(start)
        return Expr(:block, [:(storage[$(start + j - 1)] = terms[$j]) for j in 1:K]...)
    end
    last_start = N - K + 1
    body = stores(last_start)
    for start in reverse(1:(last_start - 1))
        body = :(at == Int32($start) ? $(stores(start)) : $body)
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $body
        end
        at + Int32($K)
    end
end
