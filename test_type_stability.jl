# Test type stability of process_hit_and_generate_shadow_rays!

using Revise
using Hikari
using Hikari.Raycore
using Test

# Create mock hit queue
struct MockHitQueue
    hit_found::Vector{Bool}
    tri::Vector{Triangle}
    dist::Vector{Float32}
    bary::Vector{Vec3f}
    ray::Vector{Ray}
end

Base.length(q::MockHitQueue) = length(q.hit_found)

# Create mock shadow ray queue
mutable struct MockShadowRayQueue
    ray::Vector{Ray}
    hit_idx::Vector{Int32}
    light_idx::Vector{Int32}
end

# Create test data
function setup_test_data()
    # Create a simple triangle
    tri = Triangle(
        Point3f(0, 0, 0),
        Point3f(1, 0, 0),
        Point3f(0, 1, 0),
        Normal3f(0, 0, 1),
        Normal3f(0, 0, 1),
        Normal3f(0, 0, 1),
        Point2f(0, 0),
        Point2f(1, 0),
        Point2f(0, 1),
        Int32(1)
    )

    ray = Ray(o=Point3f(0.5f0, 0.5f0, -1.0f0), d=Vec3f(0, 0, 1))

    hit_queue = MockHitQueue(
        [true],
        [tri],
        [1.0f0],
        [Vec3f(0.33f0, 0.33f0, 0.34f0)],
        [ray]
    )

    # Create lights tuple
    lights = (
        Hikari.PointLight(Point3f(2, 2, 2), Hikari.RGBSpectrum(1.0f0, 1.0f0, 1.0f0)),
        Hikari.AmbientLight(Hikari.RGBSpectrum(0.1f0, 0.1f0, 0.1f0))
    )

    nlights = length(lights)
    shadow_ray_queue = MockShadowRayQueue(
        fill(Ray(o=Point3f(0,0,0), d=Vec3f(0,0,1)), nlights),
        fill(Int32(0), nlights),
        fill(Int32(0), nlights)
    )

    return hit_queue, shadow_ray_queue, lights
end

println("Setting up test data...")
hit_queue, shadow_ray_queue, lights = setup_test_data()

println("\nTesting type stability of process_hit_and_generate_shadow_rays!...")

# Test with @code_warntype
using InteractiveUtils

idx = Int32(1)
nlights = Val{2}()

println("\n" * "="^80)
println("Code inspection for process_hit_and_generate_shadow_rays!")
println("="^80)

@code_warntype Hikari.process_hit_and_generate_shadow_rays!(
    hit_queue,
    shadow_ray_queue,
    lights,
    idx,
    nlights
)

println("\n" * "="^80)
println("Running @inferred test...")
println("="^80)

# Test that the function is type-stable
try
    @inferred Hikari.process_hit_and_generate_shadow_rays!(
        hit_queue,
        shadow_ray_queue,
        lights,
        idx,
        nlights
    )
    println("✓ Function is type-stable (no inference errors)")
catch e
    println("✗ Function has type instability:")
    println(e)
end

println("\n" * "="^80)
println("Running allocation test (should be zero allocations)...")
println("="^80)

# Warm up
for _ in 1:10
    Hikari.process_hit_and_generate_shadow_rays!(
        hit_queue,
        shadow_ray_queue,
        lights,
        idx,
        nlights
    )
end

# Measure allocations
allocs = @allocated Hikari.process_hit_and_generate_shadow_rays!(
    hit_queue,
    shadow_ray_queue,
    lights,
    idx,
    nlights
)

println("Allocations: $allocs bytes")
if allocs == 0
    println("✓ Zero allocations - function is fully optimized")
else
    println("✗ Found allocations - potential performance issue")
end

nothing
