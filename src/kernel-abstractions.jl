import KernelAbstractions as KA
import Adapt

# ============================================================================
# Adapt.adapt_structure methods for GPU conversion
# ============================================================================

# Texture - adapt data array
function Adapt.adapt_structure(to, m::Hikari.Texture)
    Hikari.Texture(Adapt.adapt(to, m.data))
end

# HomogeneousMedium - already bitstype, no adaptation needed
Adapt.adapt_structure(to, m::Hikari.HomogeneousMedium) = m

# MajorantGrid - adapt voxels array
function Adapt.adapt_structure(to, m::Hikari.MajorantGrid)
    Hikari.MajorantGrid(Adapt.adapt(to, m.voxels), m.res)
end

# GridMedium - adapt density and majorant_grid
function Adapt.adapt_structure(to, m::Hikari.GridMedium)
    Hikari.GridMedium(
        m.bounds,
        m.render_to_medium,
        m.medium_to_render,
        m.σ_a,
        m.σ_s,
        Adapt.adapt(to, m.density),
        m.density_res,
        m.g,
        Adapt.adapt(to, m.majorant_grid),
        m.max_density
    )
end

# RGBGridMedium - adapt optional grids and majorant_grid
function Adapt.adapt_structure(to, m::Hikari.RGBGridMedium)
    Hikari.RGBGridMedium(
        m.bounds,
        m.render_to_medium,
        m.medium_to_render,
        isnothing(m.σ_a_grid) ? nothing : Adapt.adapt(to, m.σ_a_grid),
        isnothing(m.σ_s_grid) ? nothing : Adapt.adapt(to, m.σ_s_grid),
        m.sigma_scale,
        isnothing(m.Le_grid) ? nothing : Adapt.adapt(to, m.Le_grid),
        m.Le_scale,
        m.grid_res,
        m.g,
        Adapt.adapt(to, m.majorant_grid)
    )
end

# NanoVDBMedium - adapt buffer and majorant_grid
function Adapt.adapt_structure(to, m::Hikari.NanoVDBMedium)
    Hikari.NanoVDBMedium(
        Adapt.adapt(to, m.buffer),
        m.root_offset,
        m.upper_offset,
        m.lower_offset,
        m.leaf_offset,
        m.leaf_count,
        m.lower_count,
        m.upper_count,
        m.root_table_size,
        m.inv_mat,
        m.vec,
        m.bounds,
        m.index_bbox_min,
        m.index_bbox_max,
        m.σ_a,
        m.σ_s,
        m.g,
        Adapt.adapt(to, m.majorant_grid),
        m.max_density
    )
end

# Distribution2D - adapt all arrays
function Adapt.adapt_structure(to, d::Hikari.Distribution2D)
    Hikari.Distribution2D(
        Adapt.adapt(to, d.conditional_func),
        Adapt.adapt(to, d.conditional_cdf),
        Adapt.adapt(to, d.conditional_func_int),
        Adapt.adapt(to, d.marginal_func),
        Adapt.adapt(to, d.marginal_cdf),
        d.marginal_func_int,
        d.nu,
        d.nv
    )
end

# EnvironmentMap - adapt data and distribution
function Adapt.adapt_structure(to, env::Hikari.EnvironmentMap)
    Hikari.EnvironmentMap(
        Adapt.adapt(to, env.data),
        env.rotation,
        Adapt.adapt(to, env.distribution)
    )
end

# EnvironmentLight - adapt env_map
function Adapt.adapt_structure(to, light::Hikari.EnvironmentLight)
    Hikari.EnvironmentLight(
        Adapt.adapt(to, light.env_map),
        light.scale
    )
end

# Bitstype lights - no adaptation needed
Adapt.adapt_structure(to, light::Hikari.PointLight) = light
Adapt.adapt_structure(to, light::Hikari.AmbientLight) = light
Adapt.adapt_structure(to, light::Hikari.DirectionalLight) = light
Adapt.adapt_structure(to, light::Hikari.SunLight) = light



# A `Film` has no `adapt_structure`, deliberately: `Adapt.adapt(backend,
# ::Array)` allocates, so `adapt(backend, film)` would be the film's ALLOCATOR,
# reached through an interface with nowhere to put an allocator, and the result
# would have no owner. `Film(backend, film)` is that path.
#
# Nothing else wants it: kernels take `film.framebuffer` and `film.normal`, not
# the film.
#
# This method exists to make that spelling LOUD. `Adapt.adapt` falls back to
# identity for a type with no `adapt_structure`, so with no method at all
# `adapt(backend, film)` quietly returns the HOST film, and a render then runs a
# device scene against host arrays: a failure somewhere else entirely, or not at
# all.
Adapt.adapt_structure(::Any, ::Film) = throw(ArgumentError(
    "`Adapt.adapt` does not allocate a Film — it converts, and allocating a " *
    "film's layers on a backend needs an allocator to own them. Use " *
    "`Film(backend, film)`, and `free!(film)` when done."))

