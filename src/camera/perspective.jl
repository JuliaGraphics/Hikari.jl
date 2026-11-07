struct ProjectiveCamera <: Camera
    core::CameraCore
    camera_to_screen::Transformation
    raster_to_camera::Transformation
    screen_to_raster::Transformation
    raster_to_screen::Transformation

    lens_radius::Float32
    focal_distance::Float32
    """Camera space to the image's window, 0 to 1 from the bottom left (homogeneous)."""
    camera_to_window::Mat4f

    function ProjectiveCamera(
            camera_to_world::Transformation, camera_to_screen::Transformation,
            screen_window::Bounds2,
            shutter_open::Float32, shutter_close::Float32,
            lens_radius::Float32, focal_distance::Float32,
            film::Film,
        )
        core = CameraCore(camera_to_world, shutter_open, shutter_close)
        # Computer projective camera transformations.
        resolution = scale(film.resolution..., 1)
        window_width = screen_window.p_max .- screen_window.p_min
        inv_bounds = scale((1f0 ./ window_width)..., 1)

        offset = translate(Vec3f(
            (-screen_window.p_min)..., 0f0,
        ))

        screen_to_raster = resolution * inv_bounds * offset
        raster_to_screen = inv(offset) * inv(inv_bounds) * inv(resolution)
        raster_to_camera = inv(camera_to_screen) * raster_to_screen

        new(
            core,
            camera_to_screen, raster_to_camera,
            screen_to_raster, raster_to_screen,
            lens_radius, focal_distance,
            (inv_bounds * offset * camera_to_screen).m,
        )
    end
end

struct PerspectiveCamera <: Camera
    core::ProjectiveCamera
    """
    Precomputed change of rays as we shift pixels on the plane in x-direction.
    """
    dx_camera::Vec3f
    """
    Precomputed change of rays as we shift pixels on the plane in y-direction.
    """
    dy_camera::Vec3f
    A::Float32

    """
    - `screen_window::Bounds2`: Screen space extent of the image.
    - `near`/`far`: projection clip planes. They only shape the perspective
      transform — no ray is clipped by them — so they are camera parameters
      rather than renderer-wide constants. Anything needing the camera-space
      plane the film maps onto must ask `raster_plane_z(camera)`, NOT assume
      `near`: the projection puts that plane at `2 * near`.
    """
    function PerspectiveCamera(
        camera_to_world::Transformation, screen_window::Bounds2,
        shutter_open::Float32, shutter_close::Float32,
        lens_radius::Float32, focal_distance::Float32,
        fov::Float32, film::Film; near::Float32 = 0.01f0, far::Float32 = 1000f0,
    )
        pc = ProjectiveCamera(
            inv(camera_to_world),
            perspective(fov, near, far),
            screen_window, shutter_open, shutter_close,
            lens_radius, focal_distance, film,
        )

        p_min = pc.raster_to_camera(Point3f(0))
        p_max = pc.raster_to_camera(Point3f(
            film.resolution[1], film.resolution[2], 0f0,
        ))
        dx_camera = pc.raster_to_camera(Point3f(1, 0, 0)) - p_min
        dy_camera = pc.raster_to_camera(Point3f(0, 1, 0)) - p_min
        p = (p_min[1:2] ./ p_min[3]) - (p_max[1:2] ./ p_max[3])
        A = abs(p[1] * p[2])

        new(pc, dx_camera, dy_camera, A)
    end
end


function PerspectiveCamera(eyepos, lookat, film; up=Vec3f(0, 1, 0), fov=55)
    screen = Hikari.Bounds2(Point2f(-1f0), Point2f(1f0))
    # An `up` along the view leaves the camera's frame undefined: the image came out black.
    dir = Vec3f(lookat) - Vec3f(eyepos)
    norm(cross(normalize(dir), normalize(Vec3f(up)))) > 1f-6 || throw(ArgumentError(
        "PerspectiveCamera: `up` = $up is parallel to the view direction $dir, which leaves no frame to " *
        "orient the image by; pass an `up` across it"))
    return PerspectiveCamera(
        Hikari.look_at(eyepos, lookat, up),
        screen, 0f0, 1f0, 0f0, 1f6, Float32(fov), film,
    )
end

# Camera effects accessors used by `vp_generate_camera_rays_kernel!` to
# decide which Sobol dimensions to compute.  Pinhole + frozen shutter is
# the common case (no DoF, no motion blur), and that's what the four
# benchmark scenes use, so the camera-ray kernel can skip the lens 2D and
# time 1D Sobol samples entirely.
camera_uses_motion_blur(c::PerspectiveCamera) = c.core.core.shutter_close > c.core.core.shutter_open
camera_uses_lens(c::PerspectiveCamera) = c.core.lens_radius > 0f0

@propagate_inbounds function generate_ray(
        camera::PerspectiveCamera, sample::CameraSample,
    )::Tuple{Ray,Float32}
    # Compute raster & camera sample positions.
    # p_film -> in pixels
    p_pixel = Point3f(sample.film..., 0f0)
    p_camera = camera.core.raster_to_camera(p_pixel)
    o = Point3f(0)
    d = normalize(p_camera)
    # Modify ray for depth of field.
    if camera.core.lens_radius > 0
        # Sample points on lens.
        p_lens = camera.core.lens_radius * concentric_sample_disk(sample.lens)
        # Compute point on plane of focus.
        # Camera looks in -z, so d[3] < 0; negate to get positive t.
        t = -camera.core.focal_distance / d[3]
        p_focus = o .+ d * t
        # Update ray for effects of lens.
        o = Point3f(p_lens[1], p_lens[2], 0f0)
        p_diff = p_focus .- o
        d = normalize(p_diff)
    end

    time = lerp(
        camera.core.core.shutter_open,
        camera.core.core.shutter_close,
        sample.time,
    )
    # TODO add medium
    ctw = camera.core.core.camera_to_world.m
    o = Raycore.transform_point(ctw, o)
    d = Raycore.transform_direction(ctw, Vec3f(d))  # Convert to Vec3f for proper direction transformation
    return Ray(d=normalize(d), o=o, time=time), 1.0f0
end
