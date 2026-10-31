@noinline surface_unpack_word(x::UInt64, ::Val{Shift}) where {Shift} = reinterpret(Float32, UInt32((x >> Shift) & 0xffffffff))
# Diagnostic SIMD result ABI: all inputs and the result cross the evaluation
# boundary as values. Sampling retains the existing method.
@inline function surface_apply(f::typeof(lobe_f), l::SurfaceStoredLobe{Looped, Mask}, wo::Vec3f, wi::Vec3f, lambda::Wavelengths, s::Scattering) where {Looped, Mask}
    result = surface_vector_evaluate_packed3(f, Val(Mask), l.kind, l.data..., wo..., wi..., lambda.lambda..., lambda.pdf..., s)
    a, b, c = result[1].value, result[2].value, result[3].value
    return SpectralRadiance((surface_unpack_word(a, Val(0)), surface_unpack_word(a, Val(32)), surface_unpack_word(b, Val(0)), surface_unpack_word(b, Val(32)))), surface_unpack_word(c, Val(0))
end
Base.@constprop :none @noinline @generated function surface_vector_evaluate_packed3(
        f, ::Val{Mask}, kind::UInt32,
        d1::Float32, d2::Float32, d3::Float32, d4::Float32, d5::Float32,
        d6::Float32, d7::Float32, d8::Float32, d9::Float32, d10::Float32,
        d11::Float32, d12::Float32, d13::Float32, d14::Float32, d15::Float32,
        o1::Float32, o2::Float32, o3::Float32,
        a1::Float32, a2::Float32, a3::Float32,
        w1::Float32, w2::Float32, w3::Float32, w4::Float32,
        p1::Float32, p2::Float32, p3::Float32, p4::Float32, s::Scattering
    ) where {Mask}
    kinds = Mask isa Bool ? [1:13;15:20] : [k for k in 1:20 if !iszero(Mask & (UInt32(1) << k))]
    Mask === true && push!(kinds, 14)
    args = f === typeof(lobe_f) ? :(Vec3f(o1, o2, o3), Vec3f(a1, a2, a3), lambda, s) :
        :(Vec3f(o1, o2, o3), Point2f(a1, a2), a3, lambda, s)
    call(k) = :(f(surface_restore(l, Val($k)), $(args.args...)))
    body = call(last(kinds))
    for k in reverse(kinds[1:(end - 1)])
        body = :(kind == $k ? $(call(k)) : $body)
    end
    return quote
        $(Expr(:meta, :noinline))
        l = SurfaceStoredLobe{false, Mask}(kind, (d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15))
        lambda = Wavelengths((w1, w2, w3, w4), (p1, p2, p3, p4))
        response, pdf = $body
        a = UInt64(reinterpret(UInt32, response.data[1])) | (UInt64(reinterpret(UInt32, response.data[2])) << 32)
        b = UInt64(reinterpret(UInt32, response.data[3])) | (UInt64(reinterpret(UInt32, response.data[4])) << 32)
        c = UInt64(reinterpret(UInt32, pdf))
        (VecElement(a), VecElement(b), VecElement(c))
    end
end
