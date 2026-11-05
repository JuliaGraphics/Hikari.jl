using Test, Hikari, Mantle, GeometryBasics, LinearAlgebra, Raycore, KernelAbstractions
using Hikari.TextureNodes

@testset "opaque Principled layout preserves counted storage" begin
    table=Hikari.get_srgb_table()
    lambda=Hikari.Wavelengths((450f0,510f0,570f0,630f0),(0.25f0,0.25f0,0.25f0,0.25f0))
    n=Vec3f(0,0,1);wo=normalize(Vec3f(0.2,0.1,1))
    tfc=Hikari.TextureFilterContext(Point2f(0.3,0.6),Point3f(0.3,-0.2,0.7),n)
    sc=Hikari.ShaderContext(tfc,wo,Vec3f(1f-3,0,0),Vec3f(0,1f-3,0),Vec3f(1,0,0),n,1f0,Vec3f(0),Point2f(0))
    variants=(
        (;metallic=0.35,coat_weight=0.4,sheen_weight=0.25),
        (;metallic=0.35,transmission_dispersion_scale=0.5),
        (;metallic=0.35,emission_color=(0.2,0.4,0.7),emission_strength=2),
        (;metallic=0.35,alpha=0.4),
        (;metallic=0.35,coat_weight=0.8,normal=Vec3f(0.1,0.2,1),coat_normal=Vec3f(-0.2,0.3,1)),
        (;metallic=0.35,base_color=noise(xyz()).color,roughness=wave(xyz()).fac,sheen_weight=0.4),
        (;metallic=0.35,anisotropic=0.8,anisotropic_rotation=0.3,tangent=Vec3f(1,1,0),diffuse_roughness=0.4),
        (;metallic=0.35,thin_film_thickness=400,coat_weight=0.4,coat_tint=(1,0.7,0.6)),
        (;metallic=0.35,roughness=0.001,sheen_weight=2,coat_weight=0.8),
    )
    for distribution in (:multi_ggx,:ggx,:beckmann), inputs in variants
        material=Hikari.Principled(;inputs...,distribution)
        store=Raycore.MultiTypeSet(KernelAbstractions.CPU())
        mat=Hikari.to_device_material(store,Hikari.surface_material(material))
        textures=Raycore.StaticMultiTypeSet((),Tuple(T[a for a in store.texture_gpu_arrays if typeof(a)===T] for T in store.texture_order))
        prog=Hikari.shader_programs(textures,mat.program)[mat.program.idx]
        M=Hikari.bsdf_nodes(typeof(mat))
        generic=Hikari.bsdf_kernel_nodes(M,Mantle.VulkanAPI())
        _,looped,mask,emissive=generic.parameters
        P=Hikari.SurfaceProgramNodes{5,looped,mask|Hikari.SURFACE_OPAQUE_STORAGE,emissive}
        R=Hikari.SurfaceProgramNodes{8,looped,mask|Hikari.SURFACE_OPAQUE_STORAGE,emissive}
        for regularize in (false,true)
            s,start=Hikari.run_values(prog,textures,sc)
            lut=Hikari.LobeSetup(Hikari.lookup_tables(textures),regularize)
            args=(P,s,prog,sc,lut,table,lambda,start)
            reference=Hikari.surface_program_bsdf(R,s,prog,sc,lut,table,lambda,start)
            actual=Hikari.surface_program_bsdf(args...)
            @test reference.lobes isa Hikari.SurfaceLobeSpan
            @test actual.lobes isa Tuple
            @test isbitstype(typeof(actual))
            @test actual.total≈reference.total
            @test all(isapprox.(actual.emission.data,reference.emission.data))
            @test !actual.dispersive && !reference.dispersive
            for restriction in (Hikari.AnyScattering(),Hikari.OnlyReflection(),Hikari.OnlyTransmission())
                @test Hikari.share(actual.lobes,restriction)≈Hikari.share(reference.lobes,restriction)
                for wi in (normalize(Vec3f(-0.3,0.2,1)),normalize(Vec3f(-0.3,0.2,-1)))
                    a,ap=Hikari.mixture(actual.lobes,wo,wi,n,lambda,restriction)
                    b,bp=Hikari.mixture(reference.lobes,wo,wi,n,lambda,restriction)
                    @test all(isapprox.(a.data,b.data))
                    @test ap≈bp
                end
                for r in (0f0,0.1f0,0.5f0,prevfloat(1f0)),u in (Point2f(0.1,0.2),Point2f(0.55,0.75))
                    a=Hikari.pick_sample(actual.lobes,r*Hikari.share(actual.lobes,restriction),0f0,wo,u,lambda,restriction)
                    b=Hikari.pick_sample(reference.lobes,r*Hikari.share(reference.lobes,restriction),0f0,wo,u,lambda,restriction)
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
