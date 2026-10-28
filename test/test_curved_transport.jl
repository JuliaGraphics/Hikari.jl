using Test, Hikari, Raycore, GeometryBasics, Mantle

# Pure transport-state tests, not a second renderer or an image reference.
struct CurvedTransportFixture <: Hikari.Medium
    empty::Bool
end

@testset "Ordinary emitting medium still renders on the GPU" begin
    backend=Mantle.defaultbackend()
    scene=Hikari.Scene(;backend)
    fog=Hikari.HomogeneousMedium(σ_a=Hikari.RGBSpectrum(0.2f0),
        σ_s=Hikari.RGBSpectrum(0f0),Le=Hikari.RGBSpectrum(2f0))
    push!(scene,GeometryBasics.normal_mesh(GeometryBasics.Tesselation(Sphere(Point3f(0),1f0),16)),
        Hikari.MediumInterface(Hikari.NullMaterial();inside=fog))
    Hikari.sync!(scene)
    function frame()
        film=Hikari.Film(Point2f(16,16))
        cam=Hikari.PerspectiveCamera(Point3f(0,-4,0),Point3f(0),film;up=Vec3f(0,0,1),fov=40f0)
        film=Hikari.Film(backend,film);Hikari.clear!(film)
        film,cam
    end
    fa,ca=frame();va=Hikari.VolPath(samples=4,max_depth=8);va(scene,fa,ca)
    fb,cb=frame();vb=Hikari.VolPath(samples=1,max_depth=8)
    for _ in 1:4
        Hikari.render!(vb,scene,fb,cb)
    end
    a=Array(fa.framebuffer);b=Array(fb.framebuffer)
    @test a==b
    @test all(c->isfinite(c.r)&&isfinite(c.g)&&isfinite(c.b),a)
    @test sum(c->c.r+c.g+c.b,a)>0
    # The long-transport scheduling lane increments the sample index ONCE,
    # preserves accumulated emission, and produces the same ordinary medium.
    fc,cc=frame();vc=Hikari.VolPath(samples=4,max_depth=128);vc(scene,fc,cc)
    @test Array(fc.framebuffer)==a
    @test vc.state.plans.sample isa Hikari.ChunkedSamplePlan
end
rotate_transport(d,a) = Vec3f(cos(a)*d[1]-sin(a)*d[2],sin(a)*d[1]+cos(a)*d[2],d[3])
Hikari.medium_step_limit(::CurvedTransportFixture,p::Point3f,d::Vec3f) = 0.25f0
Hikari.medium_start_direction(::CurvedTransportFixture,p,d,h) = rotate_transport(d,h/2)
Hikari.medium_end_direction(::CurvedTransportFixture,p,d,h,t) = rotate_transport(d,t-h/2)
Hikari.sample_point(::CurvedTransportFixture,media,table,p::Point3f,wl::Hikari.Wavelengths,d::Vec3f) =
    Hikari.MediumProperties(Hikari.SpectralRadiance(0),Hikari.SpectralRadiance(0),Hikari.SpectralRadiance(0),0f0)
function Hikari.create_majorant_iterator(m::CurvedTransportFixture,table::Hikari.RGBToSpectrumTable,
    ray::Raycore.Ray,tmax::Float32,wl::Hikari.Wavelengths,grid::M) where {M<:Hikari.MajorantGrid}
    m.empty && return Hikari.RayMajorantIterator(grid)
    Hikari.RayMajorantIterator(Hikari.HomogeneousMajorantIterator(0f0,tmax,Hikari.SpectralRadiance(100)),grid)
end

@testset "Curved medium chord lifecycle" begin
    table=Hikari.get_srgb_table()
    wl=Hikari.sample_wavelengths_visible(0.3f0)
    key=Raycore.SetKey(UInt32(1),UInt32(1))
    ray=Raycore.Ray(o=Point3f(0),d=Vec3f(1,0,0))
    unit=Hikari.SpectralRadiance(1)
    for empty in (false,true)
        media=Raycore.StaticMultiTypeSet(([CurvedTransportFixture(empty)],),())
        physical=Hikari.VPRayWorkItem(ray,wl,Int32(1);medium_idx=key)
        chord=Hikari.prepare_medium_work(physical,media)
        @test chord.ray.d ≈ rotate_transport(ray.d,0.125f0)
        @test chord.ray.t_max==0.25f0
        work=Hikari.VPMediumSampleWorkItem(chord,Raycore.HitRecord())
        result=Hikari.sample_with_iterator_helper(media.data[1][1],table,chord.ray,Inf32,wl,
            unit,unit,unit,unit,UInt64(2),nothing,zeros(Float32,4),work,media,key,Int32(8),Hikari.EmptyMajorantGrid())
        # Even many null events stay on this ONE chord. An empty iterator must
        # advance too; neither case may retrace from the unchanged origin.
        @test result.ray_o ≈ Point3f(0.25f0*chord.ray.d)
        @test result.ray_d ≈ rotate_transport(ray.d,0.25f0)
        @test result.retrace && !result.done
        @test result.beta == unit
    end
    fog=Hikari.HomogeneousMedium()
    media=Raycore.StaticMultiTypeSet(([fog],),())
    work=Hikari.VPRayWorkItem(ray,wl,Int32(1);medium_idx=key)
    @test Hikari.prepare_medium_work(work,media) === work
    @test Hikari.medium_step_limit(fog,ray.o,ray.d) == Inf32
    @test Hikari.sample_point(fog,nothing,table,ray.o,wl,ray.d) ==
          Hikari.sample_point(fog,nothing,table,ray.o,wl)
end

# The same chord through a majorant GRID: several segments, each of null events
# only. Bending stays once per chord, at its end, whatever the voxel count.
struct CurvedGridFixture <: Hikari.Medium
    majorant_grid::Hikari.MajorantGrid{Vector{Float32}}
end
Hikari.medium_step_limit(::CurvedGridFixture,p::Point3f,d::Vec3f) = 0.25f0
Hikari.medium_start_direction(::CurvedGridFixture,p,d,h) = rotate_transport(d,h/2)
Hikari.medium_end_direction(::CurvedGridFixture,p,d,h,t) = rotate_transport(d,t-h/2)
Hikari.sample_point(::CurvedGridFixture,media,table,p::Point3f,wl::Hikari.Wavelengths,d::Vec3f) =
    Hikari.MediumProperties(Hikari.SpectralRadiance(0),Hikari.SpectralRadiance(0),Hikari.SpectralRadiance(0),0f0)
Hikari.get_template_grid(m::CurvedGridFixture) = m.majorant_grid
Hikari.create_majorant_iterator(m::CurvedGridFixture,table::Hikari.RGBToSpectrumTable,
    ray::Raycore.Ray,tmax::Float32,wl::Hikari.Wavelengths,::Hikari.MajorantGrid) =
    Hikari.RayMajorantIterator(Hikari.create_dda_iterator(m.majorant_grid,
        Hikari.Bounds3(Point3f(-1f-3,-1,-1),Point3f(0.3f0,1,1)),ray.o,ray.d,0f0,tmax,Hikari.SpectralRadiance(1)))

@testset "Curved chord through a majorant grid bends once" begin
    table=Hikari.get_srgb_table()
    wl=Hikari.sample_wavelengths_visible(0.3f0)
    key=Raycore.SetKey(UInt32(1),UInt32(1))
    ray=Raycore.Ray(o=Point3f(0),d=Vec3f(1,0,0))
    unit=Hikari.SpectralRadiance(1)
    grid=Hikari.MajorantGrid(fill(40f0,4),Vec{3,Int32}(4,1,1))
    medium=CurvedGridFixture(grid)
    media=Raycore.StaticMultiTypeSet(([medium],),())
    @test Hikari.get_template_grid_from_tuple(media) === grid
    @test Hikari.get_template_grid_from_tuple((Hikari.HomogeneousMedium(),medium)) === grid
    @test Hikari.get_template_grid_from_tuple((Hikari.HomogeneousMedium(),)) == Hikari.EmptyMajorantGrid()
    physical=Hikari.VPRayWorkItem(ray,wl,Int32(1);medium_idx=key)
    chord=Hikari.prepare_medium_work(physical,media)
    work=Hikari.VPMediumSampleWorkItem(chord,Raycore.HitRecord())
    result=Hikari.sample_with_iterator_helper(medium,table,chord.ray,Inf32,wl,
        unit,unit,unit,unit,UInt64(2),nothing,zeros(Float32,4),work,media,key,Int32(8),grid)
    @test result.ray_o ≈ Point3f(0.25f0*chord.ray.d)
    @test result.ray_d ≈ rotate_transport(ray.d,0.25f0)
    @test result.retrace && !result.done
    @test result.beta == unit
end
