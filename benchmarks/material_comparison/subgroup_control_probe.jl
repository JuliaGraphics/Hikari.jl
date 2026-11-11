module SubgroupControlProbe
using Hikari, Mantle, Revise

function pin_stage(x, width)
    x isa Expr || return x
    x == :(stage_next = C_NULL) && return :(stage_next =
        can_require_subgroup_size(ctx, $width) ?
        VK.PipelineShaderStageRequiredSubgroupSizeCreateInfo(UInt32($width)) : C_NULL)
    Expr(x.head, (pin_stage(a, width) for a in x.args)...)
end

function run(f, device; width = 32)
    Mantle.can_require_subgroup_size(device, width) || error("Requested subgroup width is unavailable")
    method = which(Mantle.get_compute_pipeline, Tuple{typeof(device), Vector{UInt8}, String})
    original = Revise.CodeTracking.definition(Expr, method)
    original isa Expr || error("Cannot recover pipeline creation")
    L = Mantle.Lava
    world = L._initialization_world[]
    try
        Core.eval(Mantle, pin_stage(original, width))
        Mantle.clear_kernel_cache!(device)
        L._initialization_world[] = Base.get_world_counter()
        Base.invokelatest(f)
    finally
        Core.eval(Mantle, original)
        Mantle.clear_kernel_cache!(device)
        L._initialization_world[] = world
    end
    nothing
end
end
