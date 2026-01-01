# ============================================================================
# Postprocessing Pipeline
# ============================================================================
# Non-destructive postprocessing: reads from film.framebuffer, writes to film.postprocess
# Can be called multiple times with different parameters without re-rendering.

using ImageCore: RGB

# ============================================================================
# Tonemapping Implementations
# ============================================================================

"""
Simple Reinhard tonemapping: L / (1 + L)
"""
@inline function _tonemap_reinhard(r::Float32, g::Float32, b::Float32)
    lum = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
    scale = ifelse(lum > 0f0, 1f0 / (1f0 + lum), 1f0)
    clamp(r * scale, 0f0, 1f0), clamp(g * scale, 0f0, 1f0), clamp(b * scale, 0f0, 1f0)
end

"""
Extended Reinhard with white point control.
"""
@inline function _tonemap_reinhard_extended(r::Float32, g::Float32, b::Float32, Lwhite::Float32)
    lum = 0.2126f0 * r + 0.7152f0 * g + 0.0722f0 * b
    Lwhite2 = Lwhite * Lwhite
    scale = ifelse(lum > 0f0, (1f0 + lum / Lwhite2) / (1f0 + lum), 1f0)
    clamp(r * scale, 0f0, 1f0), clamp(g * scale, 0f0, 1f0), clamp(b * scale, 0f0, 1f0)
end

"""
ACES filmic approximation (Narkowicz).
Industry-standard filmic curve used in games and film.
"""
@inline function _tonemap_aces(r::Float32, g::Float32, b::Float32)
    a = 2.51f0
    b_c = 0.03f0
    c = 2.43f0
    d = 0.59f0
    e = 0.14f0

    r_out = clamp((r * (a * r + b_c)) / (r * (c * r + d) + e), 0f0, 1f0)
    g_out = clamp((g * (a * g + b_c)) / (g * (c * g + d) + e), 0f0, 1f0)
    b_out = clamp((b * (a * b + b_c)) / (b * (c * b + d) + e), 0f0, 1f0)
    r_out, g_out, b_out
end

"""
Uncharted 2 filmic curve helper.
"""
@inline function _uncharted2_partial(x::Float32)
    A = 0.15f0
    B = 0.50f0
    C = 0.10f0
    D = 0.20f0
    E = 0.02f0
    F = 0.30f0
    ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
end

"""
Uncharted 2 filmic tonemapping.
Good for high-contrast scenes, preserves detail in shadows.
"""
@inline function _tonemap_uncharted2(r::Float32, g::Float32, b::Float32)
    W = 11.2f0
    exposure_bias = 2.0f0

    r_out = _uncharted2_partial(r * exposure_bias)
    g_out = _uncharted2_partial(g * exposure_bias)
    b_out = _uncharted2_partial(b * exposure_bias)

    white_scale = 1f0 / _uncharted2_partial(W)

    clamp(r_out * white_scale, 0f0, 1f0),
    clamp(g_out * white_scale, 0f0, 1f0),
    clamp(b_out * white_scale, 0f0, 1f0)
end

"""
Filmic tonemapping (Hejl-Dawson).
Alternative filmic curve with good highlight rolloff.
"""
@inline function _tonemap_filmic(r::Float32, g::Float32, b::Float32)
    # Attempt to preserve some color saturation
    function filmic_channel(x::Float32)
        x = max(0f0, x - 0.004f0)
        (x * (6.2f0 * x + 0.5f0)) / (x * (6.2f0 * x + 1.7f0) + 0.06f0)
    end
    filmic_channel(r), filmic_channel(g), filmic_channel(b)
end

# ============================================================================
# Main Postprocessing Function
# ============================================================================

"""
    postprocess!(film::Film; exposure=1.0f0, tonemap=:aces, gamma=2.2f0, white_point=4.0f0)

Apply postprocessing to film.framebuffer and write result to film.postprocess.

This function is non-destructive: the original framebuffer is preserved, allowing
you to call postprocess! multiple times with different parameters.

# Arguments
- `film`: The Film containing rendered data
- `exposure`: Exposure multiplier applied before tonemapping (default: 1.0)
- `tonemap`: Tonemapping method (default: :aces)
  - `:reinhard` - Simple Reinhard L/(1+L)
  - `:reinhard_extended` - Extended Reinhard with white point
  - `:aces` - ACES filmic (industry standard)
  - `:uncharted2` - Uncharted 2 filmic
  - `:filmic` - Hejl-Dawson filmic
  - `nothing` - No tonemapping (linear clamp)
- `gamma`: Gamma correction value (default: 2.2, use `nothing` to skip)
- `white_point`: White point for extended Reinhard (default: 4.0)

# Example
```julia
# Render once
integrator(scene, film, camera)
to_framebuffer!(film)

# Try different postprocessing settings
postprocess!(film; exposure=1.0, tonemap=:aces)
display(film.postprocess)

postprocess!(film; exposure=1.5, tonemap=:reinhard)
display(film.postprocess)  # Different look, same render
```
"""
function postprocess!(film::Film;
    exposure::Real = 1.0,
    tonemap::Union{Symbol, Nothing} = :aces,
    gamma::Union{Real, Nothing} = 2.2,
    white_point::Real = 4.0,
)
    src = film.framebuffer
    dst = film.postprocess

    # Convert to Float32
    exp_f32 = Float32(exposure)
    wp_f32 = Float32(white_point)
    inv_gamma = isnothing(gamma) ? 1.0f0 : 1f0 / Float32(gamma)
    apply_gamma = !isnothing(gamma)

    @inbounds for i in eachindex(src)
        c = src[i]

        # Apply exposure
        r = c.r * exp_f32
        g = c.g * exp_f32
        b = c.b * exp_f32

        # Apply tonemapping
        if tonemap === :reinhard
            r, g, b = _tonemap_reinhard(r, g, b)
        elseif tonemap === :reinhard_extended
            r, g, b = _tonemap_reinhard_extended(r, g, b, wp_f32)
        elseif tonemap === :aces
            r, g, b = _tonemap_aces(r, g, b)
        elseif tonemap === :uncharted2
            r, g, b = _tonemap_uncharted2(r, g, b)
        elseif tonemap === :filmic
            r, g, b = _tonemap_filmic(r, g, b)
        else
            # Linear clamp
            r = clamp(r, 0f0, 1f0)
            g = clamp(g, 0f0, 1f0)
            b = clamp(b, 0f0, 1f0)
        end

        # Apply gamma correction
        if apply_gamma
            r = r^inv_gamma
            g = g^inv_gamma
            b = b^inv_gamma
        end

        dst[i] = RGB{Float32}(r, g, b)
    end

    return film.postprocess
end
