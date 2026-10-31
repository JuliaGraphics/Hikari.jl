abstract type AccelPrimitive <: Primitive end

struct BVHPrimitiveInfo
    primitive_number::UInt32
    bounds::Bounds3
    centroid::Point3f

    function BVHPrimitiveInfo(primitive_number::Integer, bounds::Bounds3)
        new(
            primitive_number, bounds,
            0.5f0 * bounds.p_min + 0.5f0 * bounds.p_max,
        )
    end
end

struct BVHNode
    bounds::Bounds3
    children::Tuple{Maybe{BVHNode},Maybe{BVHNode}}
    split_axis::UInt8
    offset::UInt32
    n_primitives::UInt32

    """
    Construct leaf node.
    """
    function BVHNode(offset::Integer, n_primitives::Integer, bounds::Bounds3)
        new(bounds, (nothing, nothing), 0, offset, n_primitives)
    end
    """
    Construct intermediary node.
    """
    function BVHNode(axis::Integer, left::BVHNode, right::BVHNode)
        new(left.bounds ∪ right.bounds, (left, right), axis, 0, 0)
    end
end

abstract type LinearNode end
struct LinearBVHLeaf <: LinearNode
    bounds::Bounds3
    primitives_offset::UInt32
    n_primitives::UInt32
end
struct LinearBVHInterior <: LinearNode
    bounds::Bounds3
    second_child_offset::UInt32
    split_axis::UInt8
end
const LinearBVH = Union{LinearBVHLeaf,LinearBVHInterior}

struct BVHAccel <: AccelPrimitive
    primitives::Vector{P} where P<:Primitive
    max_node_primitives::UInt8
    nodes::Vector{LinearBVH}

    function BVHAccel(
        primitives::Vector{P}, max_node_primitives::Integer = 1,
    ) where P<:Primitive
        max_node_primitives = min(255, max_node_primitives)
        length(primitives) == 0 && return new(primitives, max_node_primitives)

        primitives_info = [
            BVHPrimitiveInfo(i, world_bound(p))
            for (i, p) in enumerate(primitives)
        ]

        total_nodes = Ref(0)
        ordered_primitives = Vector{P}(undef, 0)
        root = _init(
            primitives, primitives_info, 1, length(primitives),
            total_nodes, ordered_primitives, max_node_primitives,
        )

        offset = Ref{UInt32}(1)
        flattened = Vector{LinearBVH}(undef, total_nodes[])
        _unroll(flattened, root, offset)
        @real_assert total_nodes[] + 1 == offset[]

        new(ordered_primitives, max_node_primitives, flattened)
    end
end

mutable struct BucketInfo
    count::UInt32
    bounds::Bounds3
end

function _init(
    primitives::Vector{P}, primitives_info::Vector{BVHPrimitiveInfo},
    from::Integer, to::Integer, total_nodes::Ref{Int64},
    ordered_primitives::Vector{P}, max_node_primitives::Integer,
) where P<:Primitive
    total_nodes[] += 1
    n_primitives = to - from + 1
    # Compute bounds for all primitives in BVH node.
    bounds = mapreduce(
        i -> primitives_info[i].bounds, ∪, from:to, init = Bounds3(),
    )
    @inline function _create_leaf()::BVHNode
        first_offset = length(ordered_primitives) + 1
        for i in from:to
            push!(
                ordered_primitives,
                primitives[primitives_info[i].primitive_number],
            )
        end
        return BVHNode(first_offset, n_primitives, bounds)
    end

    n_primitives == 1 && return _create_leaf()
    # Compute bound of primitive centroids, choose split dimension.
    centroid_bounds = mapreduce(
        i -> Bounds3(primitives_info[i].centroid), ∪, from:to,
        init = Bounds3(),
    )
    dim = maximum_extent(centroid_bounds)
    ( # Create leaf node.
        !is_valid(centroid_bounds)
        ||
        centroid_bounds.p_min[dim] == centroid_bounds.p_max[dim]
    ) && return _create_leaf()
    # Partition primitives into sets and build children.
    if n_primitives <= 2 # Equally-sized subsets.
        mid = (from + to) ÷ 2
        pmid = mid > from ? mid - from + 1 : 1
        partialsort!(
            @view(primitives_info[from:to]), pmid, by = i -> i.centroid[dim],
        )
    else # Perform Surface-Area-Heuristic partitioning.
        n_buckets = 12
        buckets = [BucketInfo(0, Bounds3(Point3f(0f0))) for _ in 1:n_buckets]
        # Initialize buckets.
        for i in from:to
            b = Int32(floor(n_buckets * offset(
                centroid_bounds, primitives_info[i].centroid,
            )[dim])) + 1
            (b == n_buckets + 1) && (b -= 1)
            buckets[b].count += 1
            buckets[b].bounds = buckets[b].bounds ∪ primitives_info[i].bounds
        end
        # Compute costs for splitting after each bucket.
        costs = Vector{Float32}(undef, n_buckets - 1)
        for i in 1:(n_buckets-1)
            it1, it2 = 1:i, (i+1):(n_buckets-1)
            s1, s2 = 0, 0
            if length(it1) > 0
                s1 = length(it1) * surface_area(
                    mapreduce(b -> buckets[b].bounds, ∪, it1),
                )
            end
            if length(it2) > 0
                s2 = length(it2) * surface_area(
                    mapreduce(b -> buckets[b].bounds, ∪, it2),
                )
            end
            costs[i] = 1f0 + (s1 + s2) / surface_area(bounds)
        end
        # Find bucket to split that minimizes SAH metric.
        min_cost_id = argmin(costs)
        leaf_cost = n_primitives
        # Either create leaf or split primitives at selected SAH bucket.
        !(
            n_primitives > max_node_primitives
            ||
            costs[min_cost_id] < leaf_cost
        ) && return _create_leaf()
        mid = partition!(primitives_info, from:to, i -> begin
            b = Int32(floor(
                n_buckets * offset(centroid_bounds, i.centroid)[dim],
            )) + 1
            (b == n_buckets + 1) && (b -= 1)
            b <= min_cost_id
        end)
    end
    BVHNode(
        dim,
        _init(
            primitives, primitives_info, from, mid,
            total_nodes, ordered_primitives, max_node_primitives,
        ),
        _init(
            primitives, primitives_info, mid + 1, to,
            total_nodes, ordered_primitives, max_node_primitives,
        ),
    )
end

function _unroll(
    linear_nodes::Vector{LinearBVH}, node::BVHNode, offset::Ref{UInt32},
)
    l_offset = offset[]
    offset[] += 1

    if node.children[1] isa Nothing
        linear_nodes[l_offset] = LinearBVHLeaf(
            node.bounds, node.offset, node.n_primitives,
        )
        return l_offset + 1
    end

    _unroll(linear_nodes, node.children[1], offset)
    second_child_offset = _unroll(linear_nodes, node.children[2], offset) - 1
    linear_nodes[l_offset] = LinearBVHInterior(
        node.bounds, second_child_offset, node.split_axis,
    )
    l_offset + 1
end

@inline function world_bound(bvh::BVHAccel)::Bounds3
    length(bvh.nodes) > 0 ? bvh.nodes[1].bounds : Bounds3()
end

function intersect!(bvh::BVHAccel, ray::AbstractRay)
    hit = false
    interaction::Maybe{SurfaceInteraction} = nothing
    length(bvh.nodes) == 0 && return hit, interaction

    check_direction!(ray)
    inv_dir = 1f0 ./ ray.d
    dir_is_neg = is_dir_negative(ray.d)

    to_visit_offset, current_node_i = 1, 1
    nodes_to_visit = zeros(Int32, 64)

    while true
        ln = bvh.nodes[current_node_i]
        if intersect_p(ln.bounds, ray, inv_dir, dir_is_neg)
            if ln isa LinearBVHLeaf && ln.n_primitives > 0
                # Intersect ray with primitives in node.
                for i in 0:ln.n_primitives-1
                    tmp_hit, tmp_interaction = intersect!(
                        bvh.primitives[ln.primitives_offset+i], ray,
                    )
                    if tmp_hit
                        hit = tmp_hit
                        interaction = tmp_interaction
                    end
                end
                to_visit_offset == 1 && break
                to_visit_offset -= 1
                current_node_i = nodes_to_visit[to_visit_offset]
            else
                if dir_is_neg[ln.split_axis] == 2
                    nodes_to_visit[to_visit_offset] = current_node_i + 1
                    current_node_i = ln.second_child_offset
                else
                    nodes_to_visit[to_visit_offset] = ln.second_child_offset
                    current_node_i += 1
                end
                to_visit_offset += 1
            end
        else
            to_visit_offset == 1 && break
            to_visit_offset -= 1
            current_node_i = nodes_to_visit[to_visit_offset]
        end
    end
    hit, interaction
end

function intersect_p(bvh::BVHAccel, ray::AbstractRay)
    length(bvh.nodes) == 0 && return false

    check_direction!(ray)
    inv_dir = 1f0 ./ ray.d
    dir_is_neg = is_dir_negative(ray.d)

    to_visit_offset, current_node_i = 1, 1
    nodes_to_visit = zeros(Int32, 64)

    while true
        ln = bvh.nodes[current_node_i]
        if intersect_p(ln.bounds, ray, inv_dir, dir_is_neg)
            if ln isa LinearBVHLeaf && ln.n_primitives > 0
                for i in 0:ln.n_primitives-1
                    intersect_p(
                        bvh.primitives[ln.primitives_offset+i], ray,
                    ) && return true
                end
                to_visit_offset == 1 && break
                to_visit_offset -= 1
                current_node_i = nodes_to_visit[to_visit_offset]
            else
                if dir_is_neg[ln.split_axis] == 2
                    nodes_to_visit[to_visit_offset] = current_node_i + 1
                    current_node_i = ln.second_child_offset
                else
                    nodes_to_visit[to_visit_offset] = ln.second_child_offset
                    current_node_i += 1
                end
                to_visit_offset += 1
            end
        else
            to_visit_offset == 1 && break
            to_visit_offset -= 1
            current_node_i = nodes_to_visit[to_visit_offset]
        end
    end
    false
end
