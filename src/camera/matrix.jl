"""
    MatrixCamera <: Camera

A camera defined by view and projection matrices, matching the convention used by
Makie's `Camera` struct (OpenGL-style: camera looks along -z in camera space).

This is a fallback camera for scenes that don't use `Camera3D` (e.g. `Axis3` which
uses its own camera type but still sets view/projection matrices on the scene camera).

The view matrix is world-to-camera, and the projection matrix is camera-to-clip (NDC).
Both use OpenGL conventions (right-handed, camera looks along -z, clip z in [-1, 1]).
"""
struct MatrixCamera <: Camera
    core::CameraCore
    raster_to_camera::Transformation
    dx_camera::Vec3f
    dy_camera::Vec3f
    A::Float32
end

"""
    MatrixCamera(view::Mat4f, projection::Mat4f, resolution::Point2f)

Construct a `MatrixCamera` from Makie-style view and projection matrices.

- `view`: world-to-camera matrix (OpenGL convention, camera looks along -z)
- `projection`: camera-to-clip matrix (OpenGL perspective projection)
- `resolution`: film resolution as `Point2f(width, height)`
"""
function MatrixCamera(view::Mat4f, projection::Mat4f, resolution::Point2f)
    # camera_to_world = inv(view)
    camera_to_world = Transformation(inv(view))
    core = CameraCore(camera_to_world, 0f0, 1f0)

    # Build raster_to_camera: pixel coords → camera-space point
    # Same chain as ProjectiveCamera but using the given projection matrix.
    #
    # raster [0..W, 0..H] → screen [-1..1, -1..1] → camera (via inv(projection))
    W, H = resolution
    screen_window = Bounds2(Point2f(-1f0, -1f0), Point2f(1f0, 1f0))
    window_width = screen_window.p_max .- screen_window.p_min
    inv_bounds = scale((1f0 ./ window_width)..., 1)
    offset = translate(Vec3f((-screen_window.p_min)..., 0f0))
    res_scale = scale(W, H, 1)

    raster_to_screen = inv(offset) * inv(inv_bounds) * inv(res_scale)
    raster_to_camera = Transformation(inv(projection)) * raster_to_screen

    # Precompute ray differentials (same as PerspectiveCamera)
    p_min = raster_to_camera(Point3f(0))
    p_max = raster_to_camera(Point3f(W, H, 0f0))
    dx_camera = raster_to_camera(Point3f(1, 0, 0)) - p_min
    dy_camera = raster_to_camera(Point3f(0, 1, 0)) - p_min
    p = (p_min[1:2] ./ p_min[3]) - (p_max[1:2] ./ p_max[3])
    A = abs(p[1] * p[2])

    MatrixCamera(core, raster_to_camera, dx_camera, dy_camera, A)
end

"""
    MatrixCamera(view, projection, resolution, screen_window)

Construct a `MatrixCamera` with a custom screen window (NDC sub-region).
Used for rendering a cropped viewport where film pixels correspond to a
sub-region of the full projection.

- `screen_window`: NDC bounds for the visible region (default full: [-1,-1] to [1,1])
"""
function MatrixCamera(view::Mat4f, projection::Mat4f, resolution::Point2f, screen_window::Bounds2)
    camera_to_world = Transformation(inv(view))
    core = CameraCore(camera_to_world, 0f0, 1f0)

    W, H = resolution
    window_width = screen_window.p_max .- screen_window.p_min
    inv_bounds = scale((1f0 ./ window_width)..., 1)
    offset = translate(Vec3f((-screen_window.p_min)..., 0f0))
    res_scale = scale(W, H, 1)

    raster_to_screen = inv(offset) * inv(inv_bounds) * inv(res_scale)
    raster_to_camera = Transformation(inv(projection)) * raster_to_screen

    p_min = raster_to_camera(Point3f(0))
    p_max = raster_to_camera(Point3f(W, H, 0f0))
    dx_camera = raster_to_camera(Point3f(1, 0, 0)) - p_min
    dy_camera = raster_to_camera(Point3f(0, 1, 0)) - p_min
    p = (p_min[1:2] ./ p_min[3]) - (p_max[1:2] ./ p_max[3])
    A = abs(p[1] * p[2])

    MatrixCamera(core, raster_to_camera, dx_camera, dy_camera, A)
end

# Convenience: accept Mat4d (Makie stores matrices as Float64)
function MatrixCamera(view::AbstractMatrix, projection::AbstractMatrix, resolution)
    MatrixCamera(Mat4f(view), Mat4f(projection), Point2f(resolution))
end

@propagate_inbounds function generate_ray(
        camera::MatrixCamera, sample::CameraSample,
    )::Tuple{Ray,Float32}
    # Same ray generation as PerspectiveCamera (no DOF support)
    p_pixel = Point3f(sample.film..., 0f0)
    p_camera = camera.raster_to_camera(p_pixel)
    d = normalize(Vec3f(p_camera))
    o = Point3f(0)

    time = lerp(
        camera.core.shutter_open,
        camera.core.shutter_close,
        sample.time,
    )
    ctw = camera.core.camera_to_world
    o = ctw(o)
    d = ctw(d)
    return Ray(d=normalize(d), o=o, time=time), 1f0
end
