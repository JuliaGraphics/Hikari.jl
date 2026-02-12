abstract type Camera end

struct CameraCore
    camera_to_world::Transformation
    shutter_open::Float32
    shutter_close::Float32
end

struct CameraSample
    """
    Point on the film the ray passes through.
    """
    film::Point2f
    """
    Point on the lens the ray passes through.
    """
    lens::Point2f
    """
    Time at which the ray should sample the scene.
    Implementations should use this value to linearly interpolate between
    shutter_open & shutter_close time range.
    """
    time::Float32
    """
    Filter weight for this sample. Used when accumulating samples into pixels.
    Computed from filter importance sampling - samples at the center of the
    filter have higher weight than those at the edges.
    """
    filter_weight::Float32
end

# Convenience constructor for backwards compatibility (weight defaults to 1.0)
CameraSample(film::Point2f, lens::Point2f, time::Float32) =
    CameraSample(film, lens, time, 1.0f0)

"""
Compute the ray corresponding to a given sample.
It is IMPORTANT that the direction vector of ray is normalized.
Other parts of the system assume it to be so.

Returns generated ray & floating point that affects how much the radiance,
arriving at the film plane along generated ray, contributes to the final image.
Simple camera models can return 1, but cameras with simulated physical lenses
set this value to indicate how much light carries through the lenses,
based on their optical properties.
"""
function generate_ray(
        camera::C, sample::CameraSample,
    )::Tuple{Ray,Float32} where C<:Camera
end

"""
Same as `generate_ray`, but also computes rays for pixels shifted one pixel
in x & y directions on the film plane.
Useful for anti-aliasing textures.
"""
function generate_ray_differential(
        camera::C, sample::CameraSample,
    )::Tuple{RayDifferentials,Float32} where C<:Camera

    ray, wt = generate_ray(camera, sample)
    shifted_x = CameraSample(
        sample.film + Point2f(1f0, 0f0), sample.lens, sample.time,
    )
    shifted_y = CameraSample(
        sample.film + Point2f(0f0, 1f0), sample.lens, sample.time,
    )
    ray_x, wt_x = generate_ray(camera, shifted_x)
    ray_y, wt_y = generate_ray(camera, shifted_y)
    rayd = RayDifferentials(
        ray.o, ray.d, ray.t_max, ray.time,
        true, ray_x.o, ray_y.o, ray_x.d, ray_y.d
    )
    rayd, wt
end

include("perspective.jl")
include("matrix.jl")

# Generic accessor for camera_to_world transform (handles different nesting depths)
# PerspectiveCamera: camera.core (ProjectiveCamera) .core (CameraCore) .camera_to_world
# MatrixCamera: camera.core (CameraCore) .camera_to_world
get_camera_to_world(camera::PerspectiveCamera) = camera.core.core.camera_to_world
get_camera_to_world(camera::MatrixCamera) = camera.core.camera_to_world
