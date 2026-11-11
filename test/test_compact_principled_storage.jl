using Test, Hikari, Mantle, GeometryBasics, LinearAlgebra, Raycore, KernelAbstractions
using Hikari.TextureNodes

@testset "compact Principled layouts preserve counted storage" begin
    table = Hikari.get_srgb_table()
    lambda = Hikari.Wavelengths((450f0,510f0,570f0,630f0),(0.25f0,0.25f0,0.25f0,0.25f0))
    n = Vec3f(0,0,1); wo = normalize(Vec3f(0.2,0.1,1))
    tfc = Hikari.TextureFilterContext(Point2f(0.3,0.6),Point3f(0.3,-0.2,0.7),n)
    sc = Hikari.ShaderContext(tfc,wo,Vec3f(1f-3,0,0),Vec3f(0,1f-3,0),Vec3f(1,0,0),n,1f0,Vec3f(0),Point2f(0))
    variants = (
        (;), (;alpha=0.3), (;alpha=0.0),
        (;emission_color=(0.2,0.4,0.7),emission_strength=2),
        (;base_color=noise(xyz()).color,roughness=wave(xyz()).fac),
        (;anisotropic=0.8,anisotropic_rotation=0.3,tangent=Vec3f(1,1,0),normal=Vec3f(0.1,0.2,1)),
        (;thin_film_thickness=400),
        (;transmission_dispersion_scale=0.5),
    )
    for distribution in (:multi_ggx,:ggx,:beckmann), (capacity,metallic) in ((1,0.0),(2,0.0),(2,1.0)), inputs in variants
        material = Hikari.Principled(;inputs...,distribution,metallic,transmission_weight=capacity==1 ? 1.0 : 0.0)
        store = Raycore.MultiTypeSet(KernelAbstractions.CPU())
        mat = Hikari.to_device_material(store,Hikari.surface_material(material))
        textures = Raycore.StaticMultiTypeSet((),Tuple(T[a for a in store.texture_gpu_arrays if typeof(a)===T] for T in store.texture_order))
        prog = Hikari.shader_programs(textures,mat.program)[mat.program.idx]
        M = Hikari.bsdf_nodes(typeof(mat))
        _,looped,mask,emissive = Hikari.bsdf_kernel_nodes(M,Mantle.VulkanAPI()).parameters
        mask = capacity==1 ? (UInt32(1)<<7) | (distribution==:beckmann ? UInt32(1)<<18 : UInt32(0)) : mask|Hikari.SURFACE_OPAQUE_STORAGE
        P = Hikari.SurfaceProgramNodes{capacity,looped,mask,emissive}
        R = Hikari.SurfaceProgramNodes{8,looped,mask,emissive}
        for regularize in (false,true)
            s,start = Hikari.run_values(prog,textures,sc)
            lut = Hikari.LobeSetup(Hikari.lookup_tables(textures),regularize)
            if capacity==2 && metallic==1.0 && isempty(inputs)
                p=Hikari.principled_inputs(s,prog,start,sc)
                convention=distribution==:beckmann ? Hikari.CyclesBeckmann() : Hikari.CyclesGGX()
                layer=Hikari.principled_base_layers(Val(false),p,sc,lut,distribution==:multi_ggx,convention)
                # Once metal consumes the base, specular setup must not read
                # an albedo table for a lobe with exactly zero weight.
                absent,remaining=Hikari.principled_specular_layer(p,layer,Vec3f(0),
                    Hikari.LobeSetup(nothing,regularize),distribution==:multi_ggx,convention)
                @test absent.sample_weight==0f0
                @test remaining==Vec3f(0)
            end
            reference = Hikari.surface_program_bsdf(R,s,prog,sc,lut,table,lambda,start)
            actual = Hikari.surface_program_bsdf(P,s,prog,sc,lut,table,lambda,start)
            @test reference.lobes isa Hikari.SurfaceLobeSpan
            @test actual.lobes isa Tuple
            @test isbitstype(typeof(actual))
            @test actual.total≈reference.total
            @test all(isapprox.(actual.emission.data,reference.emission.data))
            @test actual.dispersive==reference.dispersive
            for restriction in (Hikari.AnyScattering(),Hikari.OnlyReflection(),Hikari.OnlyTransmission())
                @test Hikari.share(actual.lobes,restriction)≈Hikari.share(reference.lobes,restriction)
                for wi in (normalize(Vec3f(-0.3,0.2,1)),normalize(Vec3f(-0.3,0.2,-1)))
                    a,ap = Hikari.mixture(actual.lobes,wo,wi,n,lambda,restriction)
                    b,bp = Hikari.mixture(reference.lobes,wo,wi,n,lambda,restriction)
                    @test all(isapprox.(a.data,b.data))
                    @test ap≈bp
                end
                for r in (0f0,0.1f0,0.5f0,prevfloat(1f0)),u in (Point2f(0.1,0.2),Point2f(0.55,0.75))
                    a = Hikari.pick_sample(actual.lobes,r*Hikari.share(actual.lobes,restriction),0f0,wo,u,lambda,restriction)
                    b = Hikari.pick_sample(reference.lobes,r*Hikari.share(reference.lobes,restriction),0f0,wo,u,lambda,restriction)
                    @test a[1]≈b[1]
                    @test all(isapprox.(a[2].data,b[2].data))
                    @test a[3]≈b[3]
                    @test a[4:5]==b[4:5]
                    @test a[6]≈b[6]
                    @test a[7]≈b[7]
                    @test a[8]==b[8]
                end
            end
        end
    end
end
