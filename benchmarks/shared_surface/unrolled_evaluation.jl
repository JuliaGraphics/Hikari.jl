# Isolation variant: constant-index reads of the same bounded storage. Keep the
# numerical accumulation order of surface_mixture and reuse weighted_f.
@generated function surface_unrolled_mixture(lobes::L, wo::Vec3f, wi::Vec3f, ns::Vec3f, lambda::Wavelengths, s::Scattering) where {L <: Tuple}
    terms = [
        quote
                w = lobes[$k]
                if w.sample_weight > 0.0f0
                    fi, pi = weighted_f(w, wo, wi, ns, lambda, s)
                    f += fi
                    pdf += pi * w.sample_weight
            end
            end for k in fieldcount(L):-1:1
    ]
    return quote
        $(Expr(:meta, :inline))
        f, pdf = SpectralRadiance(0.0f0), 0.0f0
        $(terms...)
        return f, pdf
    end
end
@inline mixture(s::SurfaceLobeSpan, wo::Vec3f, wi::Vec3f, ns::Vec3f, lambda::Wavelengths, restriction::Scattering = AnyScattering()) =
    surface_unrolled_mixture(s.data, wo, wi, ns, lambda, restriction)
