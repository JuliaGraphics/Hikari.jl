include("primes.jl")

struct Distribution1D
    func::Vector{Float32}
    cdf::Vector{Float32}
    func_int::Float32

    function Distribution1D(func::Vector{Float32})
        n = length(func)
        cdf = Vector{Float32}(undef, n + 1)
        # Compute integral of step function at `xᵢ`.
        cdf[1] = 0f0
        @_inbounds for i in 2:length(cdf)
            cdf[i] = cdf[i-1] + func[i-1] / n
        end
        # Transform step function integral into CDF.
        func_int = cdf[n+1]
        if func_int ≈ 0f0
            @_inbounds for i in 2:n+1
                cdf[i] = i / n
            end
        else
            @_inbounds for i in 2:n+1
                cdf[i] /= func_int
            end
        end

        new(func, cdf, func_int)
    end
end

function sample_discrete(d::Distribution1D, u::Float32)
    # Find interval.
    # TODO replace current `find_interval` function.
    offset = findlast(i -> d.cdf[i] ≤ u, 1:length(d.cdf))
    offset = clamp(offset, 1, length(d.cdf) - 1)

    pdf = d.func_int > 0 ? d.func[offset] / (d.func_int * length(d.func)) : 0f0
    u_remapped = (u - d.cdf[offset]) / (d.cdf[offset+1] - d.cdf[offset])
    offset, pdf, u_remapped
end

function radical_inverse(base_index::Int64, a::UInt64)::Float32
    @real_assert base_index < 1024 "Limit for radical inverse is 1023"
    base_index == 0 && return reverse_bits(a) * 5.4210108624275222e-20

    base = PRIMES[base_index]
    inv_base = 1f0 / base
    reversed_digits = UInt64(0)
    inv_base_n = 1f0

    while a > 0
        next = UInt64(floor(a / base))
        digit = UInt64(a - next * base)
        reversed_digits = reversed_digits * base + digit
        inv_base_n *= inv_base
        a = next
    end
    min(reversed_digits * inv_base_n, 1f0)
end

@inline function reverse_bits(n::UInt32)::UInt32
    n = (n << 16) | (n >> 16)
    n = ((n & 0x00ff00ff) << 8) | ((n & 0xff00ff00) >> 8)
    n = ((n & 0x0f0f0f0f) << 4) | ((n & 0xf0f0f0f0) >> 4)
    n = ((n & 0x33333333) << 2) | ((n & 0xcccccccc) >> 2)
    ((n & 0x55555555) << 1) | ((n & 0xaaaaaaaa) >> 1)
end

@inline function reverse_bits(n::UInt64)::UInt64
    n0 = UInt64(reverse_bits(UInt32((n << 32) >> 32)))
    n1 = UInt64(reverse_bits(UInt32(n >> 32)))
    return (n0 << 32) | n1
end
