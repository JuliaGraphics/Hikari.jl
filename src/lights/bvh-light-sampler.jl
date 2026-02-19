# BVH Light Sampler - Spatially-aware light importance sampling
# Following pbrt-v4's BVHLightSampler (lightsamplers.h/cpp)
#
# Builds a BVH over bounded lights on CPU, uploads to GPU as flat node array.
# Provides two GPU operations:
# - bvh_sample_light(p, n, u) → (light_idx, pmf): importance-weighted sampling
# - bvh_pmf(p, n, light_idx) → pmf: PMF lookup via bit trail replay

import KernelAbstractions as KA

# ============================================================================
# BVH Node
# ============================================================================

"""
    LightBVHNode

BVH node for the light sampler. Stores unquantized LightBounds fields
plus tree connectivity. Interior nodes store child1 index; child0 is always
at `node_idx + 1` (left child immediately follows parent in array).

Simplification vs pbrt-v4: we skip CompactLightBounds quantization.
Node is ~64 bytes instead of 32, but avoids OctahedralVector/bit-packing complexity.
With 37k lights this is ~5MB — negligible on GPU.
"""
struct LightBVHNode
    # LightBounds fields (unquantized)
    bounds_min::Point3f
    bounds_max::Point3f
    w::Vec3f
    phi::Float32
    cosθ_o::Float32
    cosθ_e::Float32
    two_sided::Bool
    # Tree connectivity
    child1_or_light_idx::UInt32   # Interior: index of child1, Leaf: flat light index
    is_leaf::Bool
end

function LightBVHNode(lb::LightBounds; child1_or_light_idx::UInt32, is_leaf::Bool)
    LightBVHNode(
        lb.bounds.p_min, lb.bounds.p_max,
        lb.w, lb.phi, lb.cosθ_o, lb.cosθ_e, lb.two_sided,
        child1_or_light_idx, is_leaf,
    )
end

# ============================================================================
# Node Importance (GPU helper)
# ============================================================================

"""
    node_importance(node::LightBVHNode, p::Point3f, n::Vec3f) -> Float32

Compute importance of a BVH node at shading point `p` with normal `n`.
Operates directly on unquantized node fields.
"""
@propagate_inbounds function node_importance(node::LightBVHNode, p::Point3f, n::Vec3f)::Float32
    node.phi == 0f0 && return 0f0

    pc = (node.bounds_min + node.bounds_max) * 0.5f0
    d2 = Raycore.distance_squared(p, pc)
    d2 = max(d2, norm(Raycore.diagonal(Raycore.Bounds3(node.bounds_min, node.bounds_max))) * 0.5f0)

    wi = normalize(Vec3f(p - pc))
    cosθ_w = dot(node.w, wi)
    node.two_sided && (cosθ_w = abs(cosθ_w))
    sinθ_w = sqrt(max(0f0, 1f0 - cosθ_w^2))

    bounds = Raycore.Bounds3(node.bounds_min, node.bounds_max)
    cosθ_b = bound_subtended_directions(bounds, p).cosθ
    sinθ_b = sqrt(max(0f0, 1f0 - cosθ_b^2))

    sinθ_o = sqrt(max(0f0, 1f0 - node.cosθ_o^2))
    cosθ_x = cos_sub_clamped(sinθ_w, cosθ_w, sinθ_o, node.cosθ_o)
    sinθ_x = sin_sub_clamped(sinθ_w, cosθ_w, sinθ_o, node.cosθ_o)
    cosθp = cos_sub_clamped(sinθ_x, cosθ_x, sinθ_b, cosθ_b)

    cosθp <= node.cosθ_e && return 0f0

    imp = node.phi * cosθp / d2

    if n != Vec3f(0f0)
        cosθ_i = abs(dot(wi, n))
        sinθ_i = sqrt(max(0f0, 1f0 - cosθ_i^2))
        cosθp_i = cos_sub_clamped(sinθ_i, cosθ_i, sinθ_b, cosθ_b)
        imp *= cosθp_i
    end

    return max(imp, 0f0)
end

# ============================================================================
# GPU Sampling
# ============================================================================

"""
    bvh_sample_light(nodes, infinite_indices, num_infinite, num_bvh, p, n, u) -> (Int32, Float32)

Sample a light from the BVH using importance-weighted traversal.
Returns (flat_light_index, pmf). Returns (0, 0) on failure.

Following pbrt-v4's BVHLightSampler::Sample (lightsamplers.h:266-320).
"""
@propagate_inbounds function bvh_sample_light(
    nodes::AbstractVector{LightBVHNode},
    infinite_light_indices::AbstractVector{Int32},
    num_infinite::Int32, num_bvh::Int32,
    p::Point3f, n::Vec3f, u::Float32
)::Tuple{Int32, Float32}
    num_total = num_infinite + num_bvh
    num_total == Int32(0) && return (Int32(0), 0f0)

    # Probability split: infinite vs BVH
    has_bvh = num_bvh > Int32(0)
    p_infinite = Float32(num_infinite) / Float32(num_infinite + (has_bvh ? Int32(1) : Int32(0)))

    if num_infinite > Int32(0) && u < p_infinite
        # Sample infinite light uniformly
        u_remapped = u / p_infinite
        idx = min(floor_int32(u_remapped * Float32(num_infinite)), num_infinite - Int32(1)) + Int32(1)
        return (infinite_light_indices[idx], p_infinite / Float32(num_infinite))
    end

    # No BVH lights
    !has_bvh && return (Int32(0), 0f0)

    # Traverse BVH
    u_bvh = if num_infinite > Int32(0)
        min((u - p_infinite) / (1f0 - p_infinite), 0.99999994f0)
    else
        min(u, 0.99999994f0)
    end
    pmf = 1f0 - p_infinite
    node_idx = Int32(1)  # 1-based root

    # BVH traversal (max depth ~32 for 37k lights)
    for _ in 1:64
        node = nodes[node_idx]
        if node.is_leaf
            return (Int32(node.child1_or_light_idx), pmf)
        end

        # Interior: child0 at node_idx + 1, child1 at node.child1_or_light_idx
        child0_idx = node_idx + Int32(1)
        child1_idx = Int32(node.child1_or_light_idx)

        c0 = node_importance(nodes[child0_idx], p, n)
        c1 = node_importance(nodes[child1_idx], p, n)

        (c0 == 0f0 && c1 == 0f0) && return (Int32(0), 0f0)

        # Probabilistic child selection
        sum_c = c0 + c1
        p0 = c0 / sum_c

        if u_bvh < p0
            pmf *= p0
            u_bvh = u_bvh / p0
            node_idx = child0_idx
        else
            pmf *= (1f0 - p0)
            u_bvh = (u_bvh - p0) / (1f0 - p0)
            node_idx = child1_idx
        end
    end

    # Should not reach here
    return (Int32(0), 0f0)
end

# ============================================================================
# GPU PMF Lookup
# ============================================================================

"""
    bvh_pmf(nodes, light_to_bit_trail, num_infinite, num_bvh, p, n, light_flat_idx) -> Float32

Compute the PMF for a specific light at shading point (p, n).
Uses the bit trail to replay the BVH traversal path efficiently.

Following pbrt-v4's BVHLightSampler::PMF (lightsamplers.h:323-358).
"""
@propagate_inbounds function bvh_pmf(
    nodes::AbstractVector{LightBVHNode},
    light_to_bit_trail::AbstractVector{UInt32},
    num_infinite::Int32, num_bvh::Int32,
    p::Point3f, n::Vec3f, light_flat_idx::Int32
)::Float32
    light_flat_idx < Int32(1) && return 0f0
    has_bvh = num_bvh > Int32(0)

    # Check if this is an infinite light (sentinel value)
    bit_trail = light_to_bit_trail[light_flat_idx]
    if bit_trail == 0xFFFFFFFF
        num_infinite == Int32(0) && return 0f0
        return 1f0 / Float32(num_infinite + (has_bvh ? Int32(1) : Int32(0)))
    end

    !has_bvh && return 0f0

    p_infinite = Float32(num_infinite) / Float32(num_infinite + Int32(1))
    pmf = 1f0 - p_infinite
    node_idx = Int32(1)

    # Replay path through BVH using bit trail
    trail = bit_trail
    for _ in 1:64
        node = nodes[node_idx]
        node.is_leaf && return pmf

        child0_idx = node_idx + Int32(1)
        child1_idx = Int32(node.child1_or_light_idx)

        c0 = node_importance(nodes[child0_idx], p, n)
        c1 = node_importance(nodes[child1_idx], p, n)
        sum_c = c0 + c1
        sum_c <= 0f0 && return 0f0

        child = trail & UInt32(1)  # 0 = left (child0), 1 = right (child1)
        if child == UInt32(0)
            pmf *= c0 / sum_c
            node_idx = child0_idx
        else
            pmf *= c1 / sum_c
            node_idx = child1_idx
        end
        trail >>= 1
    end

    return pmf
end

# ============================================================================
# BVH Construction (CPU)
# ============================================================================

const BVH_NUM_BUCKETS = 12

"""SAH cost for splitting a LightBounds along dimension `dim`.
Following pbrt-v4 EvaluateCost (lightsamplers.h:383-396)."""
function _evaluate_cost(lb::LightBounds, bounds::Raycore.Bounds3, dim::Int)
    θ_o = acos(clamp(lb.cosθ_o, -1f0, 1f0))
    θ_e = acos(clamp(lb.cosθ_e, -1f0, 1f0))
    θ_w = min(θ_o + θ_e, Float32(π))
    sinθ_o = sqrt(max(0f0, 1f0 - lb.cosθ_o^2))

    M_omega = 2f0 * Float32(π) * (1f0 - lb.cosθ_o) +
              Float32(π) / 2f0 * (2f0 * θ_w * sinθ_o - cos(θ_o - 2f0 * θ_w) -
                                   2f0 * θ_o * sinθ_o + lb.cosθ_o)

    d = Raycore.diagonal(bounds)
    max_d = max(d[1], d[2], d[3])
    dim_d = d[dim]
    Kr = dim_d > 1f-10 ? max_d / dim_d : max_d / 1f-10

    return lb.phi * M_omega * Kr * Raycore.surface_area(bounds)
end

"""
    BVHLightSampler

Spatially-aware light sampler using a BVH over bounded lights.
Default light sampler in pbrt-v4 (lightsamplers.h:259-404).

Construction builds the BVH on CPU from light bounds. The resulting
node array and bit trail array are uploaded to GPU for kernel use.
"""
struct BVHLightSampler
    nodes::Vector{LightBVHNode}                # CPU node array
    light_to_bit_trail::Vector{UInt32}          # Per-light bit trail (indexed by flat light index)
    infinite_light_indices::Vector{Int32}       # Flat indices of infinite lights
    num_bvh_lights::Int32
    num_infinite_lights::Int32
end

"""
    BVHLightSampler(lights::Raycore.MultiTypeSet; scene_radius=10f0)

Build a BVHLightSampler from a MultiTypeSet of lights.
Separates infinite and bounded lights, builds BVH over bounded lights.
"""
function BVHLightSampler(lights::Raycore.MultiTypeSet; scene_radius::Float32=10f0)
    n = length(lights)
    if n == 0
        return BVHLightSampler(
            LightBVHNode[], zeros(UInt32, 0), Int32[], Int32(0), Int32(0),
        )
    end

    # Copy light data to CPU for BVH construction (one-time cost at scene setup).
    # BVH build is recursive SAH — must run on CPU. with_index would scalar-index GPU arrays.
    lights_static = Raycore.get_static(lights)
    cpu_data = map(Array, lights_static.data)
    cpu_lights = Raycore.StaticMultiTypeSet(cpu_data, ())

    # Separate bounded and infinite lights
    bvh_lights = Tuple{Int32, LightBounds}[]  # (flat_idx, bounds)
    infinite_indices = Int32[]

    for flat_idx in 1:n
        light_key = flat_to_light_index(cpu_lights, Int32(flat_idx))
        lb = Raycore.with_index(_get_light_bounds, cpu_lights, light_key, scene_radius)
        if isnothing(lb)
            push!(infinite_indices, Int32(flat_idx))
        elseif lb.phi > 0f0
            push!(bvh_lights, (Int32(flat_idx), lb))
        end
    end

    light_to_bit_trail = fill(0xFFFFFFFF, n)  # Default: infinite sentinel

    if isempty(bvh_lights)
        return BVHLightSampler(
            LightBVHNode[], light_to_bit_trail, infinite_indices,
            Int32(0), Int32(length(infinite_indices)),
        )
    end

    # Build BVH
    nodes = LightBVHNode[]
    _build_bvh!(nodes, light_to_bit_trail, bvh_lights, 1, length(bvh_lights), UInt32(0), 0)

    return BVHLightSampler(
        nodes, light_to_bit_trail, infinite_indices,
        Int32(length(bvh_lights)), Int32(length(infinite_indices)),
    )
end

# Helper for with_index dispatch to get light bounds
@inline function _get_light_bounds(light, scene_radius::Float32)
    light_bounds(light)
end

"""Recursive BVH construction with SAH splitting.
Following pbrt-v4 buildBVH (lightsamplers.cpp:135-238)."""
function _build_bvh!(
    nodes::Vector{LightBVHNode},
    light_to_bit_trail::Vector{UInt32},
    bvh_lights::Vector{Tuple{Int32, LightBounds}},
    start::Int, stop::Int,  # 1-based inclusive range [start, stop]
    bit_trail::UInt32, depth::Int
)
    count = stop - start + 1

    # Leaf case: single light
    if count == 1
        flat_idx, lb = bvh_lights[start]
        node_idx = length(nodes) + 1
        push!(nodes, LightBVHNode(lb; child1_or_light_idx=UInt32(flat_idx), is_leaf=true))
        light_to_bit_trail[flat_idx] = bit_trail
        return (node_idx, lb)
    end

    # Compute overall bounds and centroid bounds
    overall_lb = bvh_lights[start][2]
    centroid_bounds = Raycore.Bounds3(centroid(bvh_lights[start][2]))
    for i in (start + 1):stop
        overall_lb = union(overall_lb, bvh_lights[i][2])
        centroid_bounds = union(centroid_bounds, Raycore.Bounds3(centroid(bvh_lights[i][2])))
    end

    # SAH bucket splitting
    best_cost = Inf32
    best_dim = 0
    best_bucket = 0

    for dim in 1:3
        centroid_extent = centroid_bounds.p_max[dim] - centroid_bounds.p_min[dim]
        centroid_extent <= 0f0 && continue

        # Initialize buckets
        bucket_bounds = [LightBounds() for _ in 1:BVH_NUM_BUCKETS]
        bucket_counts = zeros(Int, BVH_NUM_BUCKETS)

        for i in start:stop
            c = centroid(bvh_lights[i][2])
            b = Int(floor(BVH_NUM_BUCKETS * Raycore.offset(centroid_bounds, c)[dim]))
            b = clamp(b, 0, BVH_NUM_BUCKETS - 1) + 1  # 1-based
            bucket_bounds[b] = union(bucket_bounds[b], bvh_lights[i][2])
            bucket_counts[b] += 1
        end

        # Evaluate splits between each pair of buckets
        for split in 1:(BVH_NUM_BUCKETS - 1)
            lb_below = LightBounds()
            count_below = 0
            for b in 1:split
                lb_below = union(lb_below, bucket_bounds[b])
                count_below += bucket_counts[b]
            end

            lb_above = LightBounds()
            count_above = 0
            for b in (split + 1):BVH_NUM_BUCKETS
                lb_above = union(lb_above, bucket_bounds[b])
                count_above += bucket_counts[b]
            end

            # Skip degenerate splits
            (count_below == 0 || count_above == 0) && continue

            cost = _evaluate_cost(lb_below, overall_lb.bounds, dim) +
                   _evaluate_cost(lb_above, overall_lb.bounds, dim)

            if cost < best_cost
                best_cost = cost
                best_dim = dim
                best_bucket = split
            end
        end
    end

    # Partition lights based on best split
    mid = if best_dim > 0
        centroid_extent = centroid_bounds.p_max[best_dim] - centroid_bounds.p_min[best_dim]
        # Partition: items with centroid bucket <= best_bucket go to left
        pivot = start
        for i in start:stop
            c = centroid(bvh_lights[i][2])
            b = Int(floor(BVH_NUM_BUCKETS * Raycore.offset(centroid_bounds, c)[best_dim]))
            b = clamp(b, 0, BVH_NUM_BUCKETS - 1) + 1
            if b <= best_bucket
                if i != pivot
                    bvh_lights[pivot], bvh_lights[i] = bvh_lights[i], bvh_lights[pivot]
                end
                pivot += 1
            end
        end
        # Handle degenerate case where partition didn't split
        if pivot == start || pivot > stop
            start + div(count, 2)
        else
            pivot - 1  # Last index in left partition
        end
    else
        # No good split found: midpoint split
        start + div(count, 2) - 1
    end

    # Ensure valid split
    mid = clamp(mid, start, stop - 1)

    # Reserve placeholder for interior node
    node_idx = length(nodes) + 1
    push!(nodes, LightBVHNode(
        overall_lb; child1_or_light_idx=UInt32(0), is_leaf=false,
    ))

    # Build left child (child0 = next node = node_idx + 1)
    _, lb0 = _build_bvh!(nodes, light_to_bit_trail, bvh_lights,
                          start, mid, bit_trail, depth + 1)

    # Build right child (child1 = current position)
    child1_idx = length(nodes) + 1
    _, lb1 = _build_bvh!(nodes, light_to_bit_trail, bvh_lights,
                          mid + 1, stop, bit_trail | (UInt32(1) << depth), depth + 1)

    # Update interior node with correct child1 index and merged bounds
    merged_lb = union(lb0, lb1)
    nodes[node_idx] = LightBVHNode(
        merged_lb; child1_or_light_idx=UInt32(child1_idx), is_leaf=false,
    )

    return (node_idx, merged_lb)
end

# ============================================================================
# GPU Upload
# ============================================================================

"""
    to_gpu(backend, sampler::BVHLightSampler) -> NamedTuple

Upload BVH light sampler data to GPU. Returns a NamedTuple with GPU arrays.
"""
function bvh_to_gpu(backend, sampler::BVHLightSampler)
    # Upload nodes - KA.allocate for struct arrays
    if !isempty(sampler.nodes)
        nodes_gpu = KA.allocate(backend, LightBVHNode, length(sampler.nodes))
        copyto!(nodes_gpu, sampler.nodes)
    else
        nodes_gpu = KA.allocate(backend, LightBVHNode, 1)
    end

    # Upload bit trail array
    if !isempty(sampler.light_to_bit_trail)
        bit_trail_gpu = KA.allocate(backend, UInt32, length(sampler.light_to_bit_trail))
        copyto!(bit_trail_gpu, sampler.light_to_bit_trail)
    else
        bit_trail_gpu = KA.allocate(backend, UInt32, 1)
    end

    # Upload infinite light indices
    if !isempty(sampler.infinite_light_indices)
        inf_indices_gpu = KA.allocate(backend, Int32, length(sampler.infinite_light_indices))
        copyto!(inf_indices_gpu, sampler.infinite_light_indices)
    else
        inf_indices_gpu = KA.allocate(backend, Int32, 1)
    end

    return (
        nodes = nodes_gpu,
        light_to_bit_trail = bit_trail_gpu,
        infinite_light_indices = inf_indices_gpu,
        num_bvh_lights = sampler.num_bvh_lights,
        num_infinite_lights = sampler.num_infinite_lights,
    )
end
