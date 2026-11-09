using Test, Hikari, Mantle, KernelAbstractions
import KernelAbstractions.KernelInterface as KI

function subsurface_storage_transitions!(out, source, lambda)
    i = KI.get_global_id().x
    i > size(out, 2) && return nothing
    data = Hikari.MVector{8, typeof(source)}(ntuple(_ -> source, Val(8)))
    span = Hikari.SurfaceLobeSpan(data, Int32(2))
    branch = (i - 1) % 3
    if branch == 0
        result = Hikari.no_samples(span, lambda).lobes
    elseif branch == 1
        params, _ = Hikari.picked_walk_params(span, Int32(2), Val(1))
        exited = Hikari.exit_lobe(typeof(source), params, source.n)
        result = Hikari.with_lobe(span, Val(1), exited)
    else
        result = Hikari.without_subsurface_lobes(span)
    end
    @inbounds begin
        out[1, i] = Float32(result.count)
        out[2, i] = Hikari.share(result, Hikari.AnyScattering())
        out[3, i] = Float32(result.data[1].lobe.kind)
    end
    return nothing
end

@testset "local BSDF arrays survive all subsurface transition returns" begin
    n = Hikari.Vec3f(0, 0, 1)
    lambda = Hikari.Wavelengths((450f0, 510f0, 570f0, 630f0), (0.25f0, 0.25f0, 0.25f0, 0.25f0))
    lobe = Hikari.SubsurfaceLobe(Hikari.SpectralRadiance(0.5f0), Hikari.SpectralRadiance(0.1f0),
        0.3f0, 1.4f0, 0f0, false)
    native = Hikari.Weighted(lobe, Hikari.SpectralRadiance(0.5f0), 0.5f0, n, Hikari.Vec3f(0))
    source = Hikari.surface_store(native, Val(true), Val(Hikari.SURFACE_PRINCIPLED_MASK | (UInt32(1) << 14)))
    dev = Mantle.Device(Mantle.defaultbackend())
    out = Mantle.devicearray(dev, zeros(Float32, 3, 48))
    try
        KI.@launch dev ndrange=48 subsurface_storage_transitions!(out, source, lambda)
        Mantle.waitidle(dev)
        actual = Array(out)
        for i in 1:48
            branch = (i - 1) % 3
            expected = branch == 0 ? (0f0, 0f0, 14f0) : branch == 1 ? (1f0, 1f0, 14f0) : (2f0, 0f0, 14f0)
            @test Tuple(actual[:, i]) == expected
        end
    finally
        Mantle.unsafe_free!(out)
    end
end
