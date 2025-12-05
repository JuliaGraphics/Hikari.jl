using RPRMakie

function Plastic(; kw...)
    return (type=:Uber,
        color=to_color(:yellow), reflection_color=Vec4f(1),
        reflection_weight=Vec4f(1), reflection_roughness=Vec4f(0),
        reflection_anisotropy=Vec4f(0), reflection_anisotropy_rotation=Vec4f(0),
        reflection_metalness=Vec4f(0), reflection_ior=Vec4f(1.5), refraction_weight=Vec4f(0),
        coating_weight=Vec4f(0), sheen_weight=Vec4f(0), emission_weight=Vec3f(0),
        transparency=Vec4f(0), reflection_mode=UInt(RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR),
        emission_mode=UInt(RPR.RPR_UBER_MATERIAL_EMISSION_MODE_SINGLESIDED),
        coating_mode=UInt(RPR.RPR_UBER_MATERIAL_IOR_MODE_PBR), sss_multiscatter=false,
        refraction_thin_surface=false, kw...
    )
end

function Glass(; kw...)
    return (type=:Uber,
        reflection_color=Vec4f(0.9), reflection_weight=Vec4f(1.0),
        reflection_roughness=Vec4f(0.0), reflection_anisotropy=Vec4f(0.0),
        reflection_anisotropy_rotation=Vec4f(0.0), reflection_mode=1,
        reflection_ior=Vec4f(1.5), refraction_color=Vec4f(0.9), refraction_weight=Vec4f(1.0),
        refraction_roughness=Vec4f(0.0), refraction_ior=Vec4f(1.5),
        refraction_thin_surface=false, refraction_absorption_color=Vec4f(0.6, 0.8, 0.6, 0.0),
        refraction_absorption_distance=Vec4f(150), refraction_caustics=true,
        coating_weight=Vec4f(0), sheen_weight=Vec4f(0), emission_weight=Vec4f(0),
        transparency=Vec4f(0), kw...
    )
end
begin
    glass = Glass(color=RGBf(0.8f0, 0.2f0, 0.2f0))
    plastic = Plastic(color=RGBf(0.9f0, 0.9f0, 0.9f0), reflection_ior=1.2, reflection_roughness=0.1)
    plastic_ceil = Plastic(color=RGBf(0.33, 0.63, 0.83))
    radiance = 500
    scene = Scene(size=(1024, 1024); lights=[
        AmbientLight(RGBf(0.6, 0.6, 0.6)),
        PointLight(Vec3f(4, 4, 10), RGBf(150, 150, 150)),
        PointLight(Vec3f(-3, 10, 2.5), RGBf(60, 60, 60)),
        PointLight(Vec3f(0, 3, 0.5), RGBf(40, 40, 40))
    ])
    cam3d!(scene)
    cm = scene.camera_controls
    mesh!(scene, model, material=glass)
    mini, maxi = extrema(Rect3f(decompose(Point, model)))
    floorrect = Rect3f(Vec3f(-10, mini[2], -10), Vec3f(20, -1, 20))
    mesh!(scene, floorrect, material=plastic_ceil)
    ceiling = Rect3f(Vec3f(-20, 12, -20), Vec3f(40, -1, 40))
    mesh!(scene, ceiling, material=plastic)
    center!(scene)
    update_cam!(scene, Vec3f(-1.6, 6.2, 0.2), Vec3f(-3.6, 2.5, 2.4), Vec3f(0, 1, 0))

    @time colorbuffer(scene; backend=RPRMakie, iterations=100)
end
