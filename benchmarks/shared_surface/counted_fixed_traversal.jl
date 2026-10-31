# Isolation variant: compact setup but bounded traversal, so padding is skipped
# by its zero sample weight rather than by a runtime loop limit.
@inline mixture(s::SurfaceLobeSpan, wo::Vec3f, wi::Vec3f, ns::Vec3f, lambda::Wavelengths, restriction::Scattering = AnyScattering()) =
    surface_mixture(s.data, length(s.data), wo, wi, ns, lambda, restriction)
@inline pick_sample(s::SurfaceLobeSpan, target::Float32, before::Float32, wo::Vec3f, u::Point2f, lambda::Wavelengths, restriction::Scattering = AnyScattering()) =
    surface_pick_sample(s.data, length(s.data), target, before, wo, u, lambda, restriction)
