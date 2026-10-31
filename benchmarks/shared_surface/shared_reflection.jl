# Optional experiment: reflection models share the same microfacet geometry.
# Dispatch only their Fresnel response, rather than cloning D/G and sampling
# for every Fresnel model. All response formulas stay in the existing methods.
struct SurfaceReflectionFresnel{Mask} <: FresnelModel
    kind::UInt32
    data::NTuple{15, Float32}
end
@inline function fresnel_rt(f::SurfaceReflectionFresnel{Mask}, mu::Float32, ior::Float32, lambda::Wavelengths) where {Mask}
    kind, d = f.kind, f.data
    capabilities = Mask isa Bool ? typemax(UInt32) : Mask
    if !iszero(capabilities & ((UInt32(1) << 4) | (UInt32(1) << 16)))
        kind in (4, 16) && return fresnel_rt(surface_fresnel(F82Tint{SpectralRadiance}, d), mu, ior, lambda)
    end
    if !iszero(capabilities & (UInt32(1) << 5))
        kind == 5 && return fresnel_rt(surface_fresnel(DielectricFresnel{SpectralRadiance}, d), mu, ior, lambda)
    end
    if !iszero(capabilities & ((UInt32(1) << 6) | (UInt32(1) << 17)))
        kind in (6, 17) && return fresnel_rt(surface_fresnel(SchlickFresnel{SpectralRadiance}, d, Val(:reflection)), mu, ior, lambda)
    end
    if !iszero(capabilities & ((UInt32(1) << 8) | (UInt32(1) << 19)))
        kind in (8, 19) && return fresnel_rt(surface_fresnel(PhysicalConductorFresnel{SpectralRadiance}, d), mu, ior, lambda)
    end
    return fresnel_rt(NoFresnel(SpectralRadiance(0.0f0)), mu, ior, lambda)
end
@inline function surface_reflection(l::SurfaceStoredLobe{Looped, Mask}, convention::C) where {Looped, Mask, C}
    d = l.data
    return MicrofacetLobe{:reflection, SurfaceReflectionFresnel{Mask}, C}(
        d[1], d[2], d[3], d[4], d[5], SurfaceReflectionFresnel{Mask}(l.kind, d)
    )
end
Base.@constprop :none @noinline @generated function surface_apply_fields(
        f, ::Val{Looped}, ::Val{Mask}, kind::UInt32,
        d1::Float32, d2::Float32, d3::Float32, d4::Float32, d5::Float32,
        d6::Float32, d7::Float32, d8::Float32, d9::Float32, d10::Float32,
        d11::Float32, d12::Float32, d13::Float32, d14::Float32, d15::Float32, args...
    ) where {Looped, Mask}
    kinds = Mask isa Bool ? [1:13;15:20] : [k for k in 1:20 if !iszero(Mask & (UInt32(1) << k))]
    Mask === true && push!(kinds, 14)
    ggx = Base.intersect(kinds, [3, 4, 5, 6, 8])
    beckmann = Base.intersect(kinds, [15, 16, 17, 19])
    other = setdiff(kinds, ggx, beckmann)
    cases = [(:(l.kind == $k), :(f(surface_restore(l, Val($k)), args...))) for k in other]
    isempty(ggx) || push!(cases, (:(l.kind in $(Tuple(ggx))), :(f(surface_reflection(l, CyclesGGX()), args...))))
    isempty(beckmann) || push!(cases, (:(l.kind in $(Tuple(beckmann))), :(f(surface_reflection(l, CyclesBeckmann()), args...))))
    body = last(cases)[2]
    for (condition, value) in reverse(cases[1:(end - 1)])
        body = :($condition ? $value : $body)
    end
    return quote
        $(Expr(:meta, :noinline))
        l = SurfaceStoredLobe{Looped, Mask}(kind, (d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15))
        $body
    end
end
