module SubsurfaceStorageProbe
using Hikari, Mantle, KernelAbstractions, GeometryBasics, Test, JSON3

function diagnose(span, lambda, n)
    selected = Hikari.subsurface_choice(span, 0.5f0)
    params, weight = Hikari.picked_walk_params(span, selected, Val(1))
    exit = Hikari.exit_lobe(typeof(span[1]), params, n)
    exited = Hikari.with_lobe(typeof(span), Val(1), exit)
    f, pdf = Hikari.mixture(exited, n, n, n, lambda)
    return (Float32(Hikari.subsurface_slot(typeof(span))), Hikari.subsurface_weight(span),
            Float32(selected), weight, params.radius[1], params.albedo[1], params.ior,
            Float32(exited.count), Hikari.share(exited, Hikari.AnyScattering()), f[1], pdf)
end

@kernel function probe!(out, span, lambda, n)
    result = diagnose(span, lambda, n)
    for i in 1:length(result)
        out[i] = result[i]
    end
end

function run(output; device)
    Mantle.defaultdevice!(device)
    S = Hikari.SpectralRadiance
    n = Vec3f(0, 0, 1)
    lambda = Hikari.sample_wavelengths_visible(0.17f0)
    mask = Hikari.SURFACE_PRINCIPLED_MASK | (UInt32(1) << 14)
    packed(w) = Hikari.surface_store(w, Val(true), Val(mask))
    specular = packed(Hikari.Weighted(Hikari.LambertLobe(S), S(0.03f0), 0.03f0, n, Vec3f(0)))
    walk = packed(Hikari.Weighted(Hikari.SubsurfaceLobe{S}(S(1f0), S(0.1f0), 0.16f0, 1.45f0, 0f0, false),
                                 S(0.97f0), 0.97f0, n, Vec3f(0)))
    span = Hikari.SurfaceLobeSpan(ntuple(i -> i == 1 ? specular : i == 2 ? walk : Hikari.blank(typeof(walk)), 8), Int32(2))
    expected = collect(diagnose(span, lambda, n))
    out = Mantle.devicearray(device, zeros(Float32, length(expected)))
    probe!(device, 1)(out, span, lambda, n; ndrange = 1)
    Mantle.waitidle(device)
    actual = Array(out)
    record = (; device = Mantle.devicename(device), expected, actual)
    mkpath(dirname(output))
    open(io -> JSON3.write(io, record), output, "w")
    println("STORAGE_CONTROL ", record)
    @test actual ≈ expected
    nothing
end
end
