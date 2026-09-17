# ============================================================================
# PBRT scene builder — PBRTScene → Hikari Scene
# ============================================================================
# Converts the parsed intermediate representation into renderable Hikari objects.

const IDENTITY4 = Mat4f(LinearAlgebra.I)

@inline function srgb_to_linear(u::Float32)
    return u <= 0.04045f0 ? u / 12.92f0 : ((u + 0.055f0) / 1.055f0)^2.4f0
end

struct PBRTResult
    scene::Scene
    camera::PerspectiveCamera
    film::Film
    integrator_settings::NamedTuple
    sensor::PixelSensor
    sensor_name::String
end

"""
    load_pbrt(filename; backend=KA.CPU(), samples=nothing, max_depth=nothing)

Load a pbrt-v4 scene file and return a ready-to-render `PBRTResult`.

# Example
```julia
r = Hikari.load_pbrt("scene.pbrt")
img = VolPath(samples=64)(r.scene, r.film, r.camera)
```
"""
function load_pbrt(filename::AbstractString;
                   backend=KA.CPU(),
                   samples::Union{Nothing, Int}=nothing,
                   max_depth::Union{Nothing, Int}=nothing,
                   hw_accel::Bool=false)
    pbrt = parse_pbrt(filename)
    build_hikari_scene(pbrt; backend=backend, samples=samples, max_depth=max_depth, hw_accel=hw_accel)
end

function build_hikari_scene(pbrt::PBRTScene;
                            backend=KA.CPU(),
                            samples::Union{Nothing, Int}=nothing,
                            max_depth::Union{Nothing, Int}=nothing,
                            hw_accel::Bool=false)
    # --- Film ---
    xres = 512
    yres = 512
    sensor_name = "cie1931"
    sensor_iso = 100f0
    sensor_wb = 0f0
    if pbrt.film !== nothing
        xres = pbrt_get_int(pbrt.film, "xresolution", 512)
        yres = pbrt_get_int(pbrt.film, "yresolution", 512)
        sensor_name = pbrt_get_string(pbrt.film, "sensor", "cie1931")
        sensor_iso = Float32(pbrt_get_float(pbrt.film, "iso", 100.0))
        sensor_wb = Float32(pbrt_get_float(pbrt.film, "whitebalance", 0.0))
    end
    exposure_time = 1f0
    if pbrt.film !== nothing
        exposure_time = Float32(pbrt_get_float(pbrt.film, "exposuretime", 1.0))
    end

    # --- Pixel Filter ---
    # pbrt-v4 default: gaussian with radius (1.5, 1.5), sigma 0.5
    pixel_filter = GaussianFilter(Point2f(1.5f0), 0.5f0)  # match pbrt-v4 default
    if pbrt.pixel_filter !== nothing
        pf = pbrt.pixel_filter
        ft = lowercase(pf.type)
        xr = Float32(pbrt_get_float(pf, "xradius", ft == "box" ? 0.5 : ft == "gaussian" ? 1.5 : 2.0))
        yr = Float32(pbrt_get_float(pf, "yradius", ft == "box" ? 0.5 : ft == "gaussian" ? 1.5 : 2.0))
        r = Point2f(xr, yr)
        if ft == "box"
            pixel_filter = BoxFilter(r)
        elseif ft == "gaussian"
            sigma = Float32(pbrt_get_float(pf, "sigma", 0.5))
            pixel_filter = GaussianFilter(r, sigma)
        elseif ft == "mitchell"
            B = Float32(pbrt_get_float(pf, haskey(pf.params, "B") ? "B" : "b", 1/3))
            C = Float32(pbrt_get_float(pf, haskey(pf.params, "C") ? "C" : "c", 1/3))
            pixel_filter = MitchellFilter(r, B, C)
        elseif ft == "triangle"
            pixel_filter = TriangleFilter(r)
        elseif ft == "sinc"
            tau = Float32(pbrt_get_float(pf, "tau", 3.0))
            pixel_filter = LanczosSincFilter(r, tau)
        end
    end
    film = Film(backend, Film(Point2f(xres, yres); filter=pixel_filter))
    sensor = PixelSensor(sensor=sensor_name, iso=sensor_iso, whitebalance=sensor_wb,
                         exposure_time=exposure_time)

    # --- Camera ---
    fov = 90f0
    lens_radius = 0f0
    focal_distance = 1f6
    if pbrt.camera !== nothing
        fov = Float32(pbrt_get_float(pbrt.camera, "fov", 90.0))
        lens_radius = Float32(pbrt_get_float(pbrt.camera, "lensradius", 0.0))
        focal_distance = Float32(pbrt_get_float(pbrt.camera, "focaldistance", 1.0e6))
    end
    # `pbrt.camera_transform` from `pbrt_lookat`, NOT `Raycore.look_at`. The two
    # differ by a `cross(up, dir)` vs `cross(dir, up)` flip, and that difference
    # is a real convention split, not a bug in either:
    #
    #   pbrt   (transform.cpp `LookAt`): right = normalize(cross(up, dir))
    #   Makie  (Raycore.look_at):        right = cross(dir, up)   [ = -pbrt's ]
    #
    # pbrt's camera space is LEFT-handed by design; Makie's is the usual
    # right-handed one. Feeding the same `LookAt` numbers to both therefore
    # produces images that are HORIZONTAL MIRRORS of each other.
    #
    # MEASURED 2026-08-20, both at 256 spp against pbrt's own EXR, mean |Δlum|:
    #
    #                              vs pbrt as-is   vs pbrt MIRRORED
    #   crown  (via pbrt_to_makie)     0.0326           0.1426     -> NOT mirrored
    #   bunny  (hand-written scene)    0.0458           0.0134     -> mirrored
    #
    # Both scenes end up in `Camera3D` -> `Raycore.look_at`, so the flip is
    # decided by the ARGUMENTS, not by the code path. They differ in how the
    # camera is derived:
    #   crown: `pbrt_to_makie` takes eye/target/up out of the CTM, giving
    #          target = eye + dir and pbrt's ORTHOGONALISED up.
    #   bunny: the .jl passes the .pbrt's raw LookAt numbers straight to
    #          `update_cam!`.
    # Which of those two inputs actually produces the flip is NOT established —
    # a cross-product argument says both should mirror, and the measurement says
    # only one does. The audit this note has always asked for is still open;
    # what is now pinned is the two data points above, so a future attempt has a
    # discriminating test (Crown is near-symmetric and cannot show a mirror —
    # that is why this note read as inconclusive for so long; use bunny-cloud).
    #
    # RayMakie is internally consistent and must stay that way: a probe with a
    # raytraced mesh and a raster `scatter!` at the same world x puts both at the
    # same column, i.e. Hikari's camera agrees with Makie's own projection. So
    # `Raycore.look_at` must NOT be "fixed" to pbrt's convention — that would
    # mirror the raytraced image away from every overlay.
    #
    # Consequence for this file: the suite keeps using `pbrt_lookat`. Anyone
    # diffing a RayMakie render against a pbrt EXR must CHECK BOTH orientations
    # rather than assume either.
    wtc_mat = pbrt.camera_transform
    ctw_mat = inv(wtc_mat)
    wtc_tf = Transformation(wtc_mat, ctw_mat)
    # pbrt-v4 cameras.cpp: screen-window aspect = width/height. For tall images
    # the screen is [-1,1] × [-1/frame, 1/frame]; for wide it's [-frame, frame]
    # × [-1, 1]. Matches RayMakie's to_trace_camera convention.
    frame = Float32(xres) / Float32(yres)
    screen = if frame >= 1f0
        Bounds2(Point2f(-frame, -1f0), Point2f(frame, 1f0))
    else
        Bounds2(Point2f(-1f0, -1f0/frame), Point2f(1f0, 1f0/frame))
    end
    camera = PerspectiveCamera(
        wtc_tf, screen, 0f0, 1f0, lens_radius, focal_distance, Float32(fov), film
    )

    # --- Integrator settings ---
    # Match pbrt-v4 defaults exactly
    int_samples = 64
    int_max_depth = 5           # pbrt-v4 default
    int_regularize = false      # pbrt-v4 wavefront default (surfscatter.cpp:196)
    int_rr_depth = 1            # pbrt-v4 hardcoded (surfscatter.cpp:215: depth >= 1)
    int_max_component = Inf32   # pbrt-v4 default (film.cpp:576: Infinity)
    if pbrt.integrator !== nothing
        int_samples = pbrt_get_int(pbrt.integrator, "pixelsamples", 64)
        int_max_depth = pbrt_get_int(pbrt.integrator, "maxdepth", 5)
        int_regularize = pbrt_get_bool(pbrt.integrator, "regularize", false)
    end
    # pbrt-v4 puts `pixelsamples` on the Sampler, not the Integrator. Crown
    # specifies `Sampler "halton" "integer pixelsamples" 512`. Without this the
    # parser fell through to the integrator default (64) and the caller's
    # `samples` arg silently overrode pbrt's scene-specified count.
    if pbrt.sampler !== nothing
        int_samples = pbrt_get_int(pbrt.sampler, "pixelsamples", int_samples)
    end
    # Film maxcomponentvalue
    if pbrt.film !== nothing
        int_max_component = Float32(pbrt_get_float(pbrt.film, "maxcomponentvalue", Inf))
    end
    samples !== nothing && (int_samples = samples)
    max_depth !== nothing && (int_max_depth = max_depth)

    # --- Build scene first ---
    # The scene owns the texture store (`scene.materials`, a MultiTypeSet), and
    # image textures must be pushed into it BEFORE materials are constructed:
    # a material's texture parameters are now `TexHandle`s carrying a runtime
    # `(slot, idx)`, which only exists once the array has been stored. Building
    # the scene here rather than after the material cache is what makes the
    # non-parametric material structs possible.
    scene = Scene(; backend=backend, hw_accel=hw_accel)

    # --- Build textures (pushed into the scene's store, returned as handles) ---
    hikari_textures = build_pbrt_textures(pbrt, scene)

    # --- Build materials cache ---
    mat_cache = Dict{String, Material}()
    for pass in (false, true)
        for (name, entity) in pbrt.named_materials
            is_mix = lowercase(entity.type) == "mix"
            is_mix == pass || continue
            haskey(mat_cache, name) && continue
            mat_cache[name] = attach_bump(
                build_pbrt_material(entity, pbrt, hikari_textures, mat_cache),
                entity, hikari_textures)
        end
    end

    # --- Build media cache ---
    media_cache = Dict{String, Medium}()
    for (name, entity) in pbrt.named_media
        transform = get(pbrt.media_transforms, name, IDENTITY4)
        med = build_pbrt_medium(entity, pbrt, transform)
        med !== nothing && (media_cache[name] = med)
    end

    # Add standalone lights
    for lrec in pbrt.lights
        light = build_pbrt_light(lrec, pbrt)
        light !== nothing && push!(scene, light)
    end

    # Add shapes with materials
    for srec in pbrt.shapes
        mesh = build_pbrt_shape(srec, pbrt)
        mesh === nothing && continue
        mat = resolve_pbrt_material(srec, mat_cache, pbrt; textures=hikari_textures)

        # Resolve media for this shape
        inside_medium = get(media_cache, srec.medium_inner, nothing)
        outside_medium = get(media_cache, srec.medium_outer, nothing)

        # Area light → wrap material in MediumInterface with Emissive
        if srec.area_light !== nothing
            emissive = Emissive(srec.area_light)
            push!(scene, mesh, MediumInterface(mat;
                emission=emissive, inside=inside_medium, outside=outside_medium))
        elseif inside_medium !== nothing || outside_medium !== nothing
            # Shape has participating media — wrap in MediumInterface
            push!(scene, mesh, MediumInterface(mat;
                inside=inside_medium, outside=outside_medium))
        else
            push!(scene, mesh, mat)
        end
    end

    sync!(scene)

    return PBRTResult(scene, camera, film,
        (samples=int_samples, max_depth=int_max_depth, regularize=int_regularize,
         russian_roulette_depth=int_rr_depth, max_component_value=int_max_component),
        sensor, sensor_name)
end

# ============================================================================
# Parameter extraction helpers
# ============================================================================

function pbrt_get_float(entity::PBRTEntity, name::String, default::Real)
    haskey(entity.params, name) || return default
    p = entity.params[name]
    isempty(p.values) && return default
    v = p.values[1]
    v isa Number || return default
    return Float64(v)
end

function pbrt_get_int(entity::PBRTEntity, name::String, default::Int)
    haskey(entity.params, name) || return default
    p = entity.params[name]
    isempty(p.values) && return default
    return Int(p.values[1])
end

function pbrt_get_string(entity::PBRTEntity, name::String, default::String)
    haskey(entity.params, name) || return default
    p = entity.params[name]
    isempty(p.values) && return default
    v = p.values[1]
    v isa AbstractString && return String(v)
    return default  # not a string value (e.g. spectrum stored as floats)
end

function pbrt_get_bool(entity::PBRTEntity, name::String, default::Bool)
    haskey(entity.params, name) || return default
    p = entity.params[name]
    isempty(p.values) && return default
    v = p.values[1]
    v isa Bool && return v
    v isa String && return lowercase(v) == "true"
    return default
end

function pbrt_get_rgb(entity::PBRTEntity, name::String, default::NTuple{3, Float64})
    haskey(entity.params, name) || return default
    p = entity.params[name]
    length(p.values) >= 3 || return default
    return (Float64(p.values[1]), Float64(p.values[2]), Float64(p.values[3]))
end

# Convert a blackbody temperature (Kelvin) to a linear sRGB triplet: take the
# chromaticity from the Planckian locus and normalize LUMINANCE to 1.
#
# DO NOT "fix" this to match pbrt-v4's `BlackbodySpectrum` normalization. That
# was tried and measured. pbrt scales its Planck spectrum so its PEAK value is 1
# (`1/Blackbody(λ_max, T)`, Wien), whose luminance is NOT 1 — 0.9796 at 5500 K,
# 0.2763 at 2700 K — so reproducing it here looks obviously more faithful. It is
# not, because pbrt is SPECTRAL: it never forms an emitter RGB at all, it carries
# wavelengths through `PixelSensor`, which applies its own imaging ratio and
# white balance. The RGB this function returns feeds Hikari's sensor pipeline,
# which expects a luminance-normalized emitter.
#
# A/B on the two blackbody suite scenes (5500 K, `scale 10`, same harness, both
# against the same pbrt reference):
#
#     formula                       smoothgold          bumpgold
#     Y normalized to 1 (this)      energy 1.0004       energy 0.9990
#     pbrt peak normalization       energy 0.9808       energy 0.9793  <- 2% dark
#
# Pinned by test_blackbody_emitter_scale.jl.
function blackbody_to_rgb(T::Float32)
    x, y = planckian_xy(T)
    X = x / y; Y = 1f0; Z = (1f0 - x - y) / y
    r =  3.2406f0 * X - 1.5372f0 * Y - 0.4986f0 * Z
    g = -0.9689f0 * X + 1.8758f0 * Y + 0.0415f0 * Z
    b =  0.0557f0 * X - 0.2040f0 * Y + 1.0570f0 * Z
    # Clamp negatives (out-of-gamut chromaticities below the spectral locus
    # can produce small negatives that wreck the radiance estimator).
    return (Float64(max(r, 0f0)), Float64(max(g, 0f0)), Float64(max(b, 0f0)))
end

"""
    Emissive(area_light::PBRTEntity)

Build the emissive material for a pbrt `AreaLightSource "diffuse"` record.

Applies pbrt-v4's photometric normalization from `lights.cpp`
`DiffuseAreaLight::Create`, which scales so radiance is equivalent to 1 nit:

    scale /= SpectrumToPhotometric(L)

Note this divides by the value for the ACTUAL emitter spectrum, not a fixed D65
constant. For an `RGBIlluminantSpectrum` the two coincide — pbrt and Hikari both
special-case it back to the colour space's illuminant — but that is a property of
that spectrum type, not a licence to hardcode D65.

Both importers (Hikari's `pbrt_build_scene` and RayMakie's `pbrt_to_makie`) go
through here so they cannot drift apart.
"""
function Emissive(area_light::PBRTEntity)
    Le = pbrt_get_emissive_le(area_light, (1.0, 1.0, 1.0))
    scale = Float32(pbrt_get_float(area_light, "scale", 1.0))
    two_sided = pbrt_get_bool(area_light, "twosided", false)
    Le_spectrum = rgb_illuminant_spectrum(get_srgb_table(),
        RGB{Float32}(Float32(Le[1]), Float32(Le[2]), Float32(Le[3])))
    scale /= spectrum_to_photometric(Le_spectrum)
    return Emissive(TexHandle(Le), scale, two_sided)
end

# Like pbrt_get_rgb but also handles "blackbody" type params (single temperature value).
# Used for emissive Le values which can be RGB or blackbody spectra.
function pbrt_get_emissive_le(entity::PBRTEntity, default::NTuple{3, Float64})
    haskey(entity.params, "L") || return default
    p = entity.params["L"]
    if p.type == :blackbody && !isempty(p.values)
        return blackbody_to_rgb(Float32(p.values[1]))
    elseif length(p.values) >= 3
        return (Float64(p.values[1]), Float64(p.values[2]), Float64(p.values[3]))
    end
    return default
end

function pbrt_get_floats(entity::PBRTEntity, name::String)
    haskey(entity.params, name) || return Float64[]
    return Float64.(entity.params[name].values)
end

function pbrt_get_ints(entity::PBRTEntity, name::String)
    haskey(entity.params, name) || return Int[]
    return Int.(entity.params[name].values)
end

# ============================================================================
# Medium building
# ============================================================================

function build_pbrt_medium(entity::PBRTEntity, pbrt::PBRTScene, transform::Mat4f=IDENTITY4)
    type = lowercase(entity.type)

    if type == "homogeneous"
        preset = pbrt_get_string(entity, "preset", "")
        default_a = (1.0, 1.0, 1.0)
        default_s = (1.0, 1.0, 1.0)
        if !isempty(preset)
            if haskey(MEDIUM_PRESETS, preset)
                p = MEDIUM_PRESETS[preset]
                default_a = (Float64(p.σ_a[1]), Float64(p.σ_a[2]), Float64(p.σ_a[3]))
                default_s = (Float64(p.σ_s[1]), Float64(p.σ_s[2]), Float64(p.σ_s[3]))
            else
                @warn "pbrt: medium preset \"$preset\" not found"
            end
        end
        sigma_a = pbrt_get_rgb(entity, "sigma_a", default_a)
        sigma_s = pbrt_get_rgb(entity, "sigma_s", default_s)
        g_val = Float32(pbrt_get_float(entity, "g", 0.0))
        return HomogeneousMedium(
            σ_a=RGBSpectrum(Float32(sigma_a[1]), Float32(sigma_a[2]), Float32(sigma_a[3])),
            σ_s=RGBSpectrum(Float32(sigma_s[1]), Float32(sigma_s[2]), Float32(sigma_s[3])),
            g=g_val)

    elseif type == "nanovdb"
        filename = pbrt_get_string(entity, "filename", "")
        isempty(filename) && (@warn "pbrt: nanovdb medium without filename"; return nothing)
        path = isabspath(filename) ? filename : joinpath(pbrt.base_dir, filename)
        isfile(path) || (@warn "pbrt: NanoVDB file not found: $path"; return nothing)

        # Parse sigma_a and sigma_s — may be "spectrum" type (wavelength/value pairs)
        # or "rgb" type. For spectrum type, use a representative value.
        sigma_a = pbrt_get_spectrum_as_rgb(entity, "sigma_a", (0.5, 0.5, 0.5))
        sigma_s = pbrt_get_spectrum_as_rgb(entity, "sigma_s", (10.0, 10.0, 10.0))
        g_val = Float32(pbrt_get_float(entity, "g", 0.0))

        # Extract 3x3 rotation from the medium's transform (set by Rotate in AttributeBegin)
        rot = Mat3f(transform[1,1], transform[2,1], transform[3,1],
                    transform[1,2], transform[2,2], transform[3,2],
                    transform[1,3], transform[2,3], transform[3,3])
        return NanoVDBMedium(path;
            σ_a=RGBSpectrum(Float32(sigma_a[1]), Float32(sigma_a[2]), Float32(sigma_a[3])),
            σ_s=RGBSpectrum(Float32(sigma_s[1]), Float32(sigma_s[2]), Float32(sigma_s[3])),
            g=g_val,
            transform=rot)

    else
        @warn "pbrt: unsupported medium type '$type'"
        return nothing
    end
end

"""Extract a spectrum parameter as RGB. Handles both "rgb" and "spectrum" (wavelength/value pairs)."""
function pbrt_get_spectrum_as_rgb(entity::PBRTEntity, name::String, default::NTuple{3,Float64})
    haskey(entity.params, name) || return default
    p = entity.params[name]
    if p.type == :rgb && length(p.values) >= 3
        return (Float64(p.values[1]), Float64(p.values[2]), Float64(p.values[3]))
    elseif p.type == :spectrum && length(p.values) >= 2
        # Wavelength/value pairs — take a representative value (average of all values)
        vals = Float64[p.values[i] for i in 2:2:length(p.values)]
        avg = sum(vals) / length(vals)
        return (avg, avg, avg)
    elseif length(p.values) >= 2 && all(v -> v isa Number, p.values)
        # Might be interleaved wavelength/value pairs stored as floats
        vals = Float64[p.values[i] for i in 2:2:length(p.values)]
        avg = sum(vals) / length(vals)
        return (avg, avg, avg)
    end
    return default
end

# ============================================================================
# Material building
# ============================================================================

# ============================================================================
# Texture building from pbrt named textures
# ============================================================================

build_pbrt_textures(pbrt::PBRTScene) = build_pbrt_textures(pbrt, nothing)

function build_pbrt_textures(pbrt::PBRTScene, scene)
    textures = Dict{String, Any}()
    # Two passes: build base textures first, then derived (scale) textures
    for pass in (false, true)
    for (name, tex_entity) in pbrt.named_textures
        tex_type = lowercase(tex_entity.type)
        is_derived = tex_type == "scale"
        is_derived == pass || continue
        tex_class = pbrt_get_string(tex_entity, "_class", "spectrum")

        if tex_type == "checkerboard"
            # Procedural — evaluated analytically at shading time via
            # CheckerboardTexture (exact pbrt-v4 port, see textures/basic.jl).
            dim = Int(pbrt_get_float(tex_entity, "dimension", 2.0))
            dim == 2 || error("pbrt: only 2D checkerboard textures are supported (got dimension=$dim)")
            uscale = Float32(pbrt_get_float(tex_entity, "uscale", 1.0))
            vscale = Float32(pbrt_get_float(tex_entity, "vscale", 1.0))
            udelta = Float32(pbrt_get_float(tex_entity, "udelta", 0.0))
            vdelta = Float32(pbrt_get_float(tex_entity, "vdelta", 0.0))
            if tex_class == "float"
                v1 = Float32(pbrt_get_float(tex_entity, "tex1", 1.0))
                v2 = Float32(pbrt_get_float(tex_entity, "tex2", 0.0))
                textures[name] = CheckerboardTexture(uscale, vscale, udelta, vdelta, v1, v2)
            else
                c1 = pbrt_get_rgb(tex_entity, "tex1", (1.0, 1.0, 1.0))
                c2 = pbrt_get_rgb(tex_entity, "tex2", (0.0, 0.0, 0.0))
                rgb1 = RGBSpectrum(Float32(c1[1]), Float32(c1[2]), Float32(c1[3]))
                rgb2 = RGBSpectrum(Float32(c2[1]), Float32(c2[2]), Float32(c2[3]))
                textures[name] = CheckerboardTexture(uscale, vscale, udelta, vdelta, rgb1, rgb2)
            end
        elseif tex_type == "constant"
            if tex_class == "float"
                val = Float32(pbrt_get_float(tex_entity, "value", 1.0))
                textures[name] = ConstTexture(val)
            else
                c = pbrt_get_rgb(tex_entity, "value", (1.0, 1.0, 1.0))
                textures[name] = ConstTexture(RGBSpectrum(Float32(c[1]), Float32(c[2]), Float32(c[3])))
            end

        elseif tex_type == "imagemap"
            filename = pbrt_get_string(tex_entity, "filename", "")
            isempty(filename) && continue
            path = isabspath(filename) ? filename : joinpath(pbrt.base_dir, filename)
            isfile(path) || (@warn "pbrt: texture image not found: $path"; continue)
            img = FileIO.load(path)
            # Normalize to RGB regardless of source channel layout (Gray, GrayA,
            # RGB, RGBA, BGR, palette, …). Earlier code called Colors.red/green/
            # blue directly on the source pixel which throws on Gray-format PNGs
            # (test_stripes.png triggers this), but `convert(RGB, gray)` is a
            # well-defined widening.
            img_rgb = convert.(Colors.RGB{Float32}, img)
            # pbrt-v4 default encoding is purely EXTENSION-based
            # (textures.cpp:436): `.png → sRGB, everything else → linear` —
            # for BOTH float and spectrum texture classes, and 16-bit PNGs are
            # decoded through the encoding just like 8-bit ones
            # (image.cpp ReadPNG: `v = encoding.ToFloatLinear(v)`). The
            # earlier rule here (`8-bit && spectrum class`) missed float
            # height maps: pbrt sRGB-decodes test_ridges.png (16-bit gray),
            # which scales the bump gradient by the local slope of the sRGB
            # curve (~1.45× at h≈0.7) — tex_conductor_bumpmap_light_point
            # pins this. Killeroo's 8-bit lines.png contrast (~4×) is also
            # covered by the same rule. Override via `"string encoding"`.
            encoding = lowercase(pbrt_get_string(tex_entity, "encoding", ""))
            default_srgb = endswith(lowercase(path), ".png")
            apply_srgb = isempty(encoding) ? default_srgb : encoding == "srgb"
            if apply_srgb
                @inbounds for idx in CartesianIndices(img_rgb)
                    px = img_rgb[idx]
                    img_rgb[idx] = Colors.RGB{Float32}(srgb_to_linear(px.r),
                                                      srgb_to_linear(px.g),
                                                      srgb_to_linear(px.b))
                end
            end
            if tex_class == "float"
                data = Matrix{Float32}(undef, size(img_rgb)...)
                for idx in CartesianIndices(img_rgb)
                    px = img_rgb[idx]
                    data[idx] = 0.2126f0 * px.r + 0.7152f0 * px.g + 0.0722f0 * px.b
                end
                textures[name] = Texture(data)
            else
                data = Matrix{RGBSpectrum}(undef, size(img_rgb)...)
                for idx in CartesianIndices(img_rgb)
                    px = img_rgb[idx]
                    data[idx] = RGBSpectrum(px.r, px.g, px.b)
                end
                textures[name] = Texture(data)
            end

        elseif tex_type == "scale"
            # scale texture: output = tex * scale. Float class multiplies the
            # underlying float data; spectrum class multiplies the underlying RGB
            # texture (used e.g. for killeroo's `sgrid = 0.5 * imagemap(lines.png)`
            # floor/wall pattern — without spectrum-scale support that texture
            # silently fell through to the default grey diffuse).
            scale_val = Float32(pbrt_get_float(tex_entity, "scale", 1.0))
            tex_ref = pbrt_get_string(tex_entity, "tex", "")
            base = isempty(tex_ref) ? nothing : get(textures, tex_ref, nothing)
            if tex_class == "float"
                if base isa Texture{Float32}
                    textures[name] = Texture(base.data .* scale_val)
                elseif base !== nothing
                    textures[name] = base
                else
                    textures[name] = ConstTexture(scale_val)
                end
            else
                if base isa Texture{RGBSpectrum}
                    scaled = Matrix{RGBSpectrum}(undef, size(base.data)...)
                    @inbounds for i in eachindex(base.data)
                        s = base.data[i]
                        scaled[i] = RGBSpectrum(s.c[1] * scale_val,
                                                s.c[2] * scale_val,
                                                s.c[3] * scale_val,
                                                s.c[4])
                    end
                    textures[name] = Texture(scaled)
                elseif base !== nothing
                    textures[name] = base
                else
                    textures[name] = ConstTexture(
                        RGBSpectrum(scale_val, scale_val, scale_val))
                end
            end
        end
    end  # for (name, tex_entity)
    end  # for pass

    # Convert every built texture into a `TexHandle`, storing the out-of-line
    # ones in the scene's texture store as we go.
    #
    # This is the step that lets material structs be non-parametric: a material
    # field holds a handle carrying a runtime `(slot, idx)`, so whether a
    # parameter is a constant, an image map or a procedural no longer changes
    # the material's TYPE — and therefore no longer multiplies closest-hit
    # shaders.
    # Without a scene there is nowhere to put them yet; the materials keep the
    # raw textures and `to_device_material` stores them at push time. That is
    # correct but stores one copy per material, so the scene-ful path — every
    # pbrt render — is the one to use when there is a choice.
    scene === nothing && return textures
    handles = Dict{String, TexHandle}()
    for (name, tex) in textures
        handles[name] = store_texture_handle(scene, tex)
    end
    return handles
end

"""
    store_texture_handle(scene, tex) -> TexHandle

Turn a built texture into a handle, pushing it into the scene's texture store
when it cannot live inline. Building the handles once here (rather than letting
each material store its own copy at push time) is what keeps a pbrt scene from
uploading the same image map once per material that references it.
"""
store_texture_handle(scene, tex) = device_param(scene.materials, tex)

"""
    attach_bump(mat, entity, textures) -> mat

Attach `entity`'s height field to `mat`'s `displacement`. pbrt-v4 master spells
it `displacement`; older `.pbrt` files use `bumpmap`. Both are float height
fields, read by finite differences at intersection time.

Public because RayMakie's pbrt bridge builds materials before any Hikari scene
exists and needs the same attachment — it called this through the underscore
when it was private, which is how a rename here became a rename there.
"""
function attach_bump(mat::Material, entity::PBRTEntity, textures::AbstractDict{String})
    for key in ("displacement", "bumpmap", "normalmap")
        haskey(entity.params, key) || continue
        p = entity.params[key]
        isempty(p.values) && continue
        tex_name = p.values[1]
        tex_name isa AbstractString || continue
        h = get(textures, String(tex_name), nothing)
        h === nothing && continue
        return set_displacement(mat, matparam(h))
    end
    return mat
end

"""Get a material parameter as a texture or constant, resolving named texture references."""
function pbrt_get_texture(entity::PBRTEntity, name::String,
                          textures::AbstractDict{String}, default_rgb)
    if haskey(entity.params, name)
        p = entity.params[name]
        if p.type == :texture && !isempty(p.values) && p.values[1] isa String
            tex_name = p.values[1]
            if haskey(textures, tex_name)
                return textures[tex_name]
            end
        end
    end
    # Fall back to constant
    return TexHandle(pbrt_get_rgb(entity, name, default_rgb))
end

function pbrt_get_float_texture(entity::PBRTEntity, name::String,
                                textures::AbstractDict{String}, default_val)
    if haskey(entity.params, name)
        p = entity.params[name]
        if !isempty(p.values) && p.values[1] isa AbstractString
            # Texture reference: look it up or use default (don't try Float64(string))
            tex_name = String(p.values[1])
            haskey(textures, tex_name) && return textures[tex_name]
            @warn "pbrt: float texture '$tex_name' not found, using default $default_val"
            return TexHandle(Float32(default_val))
        end
    end
    return TexHandle(Float32(pbrt_get_float(entity, name, Float64(default_val))))
end

function build_pbrt_material(entity::PBRTEntity, pbrt::PBRTScene,
                             textures::AbstractDict{String}=Dict{String,TexHandle}(),
                             mat_cache::Dict{String, Material}=Dict{String,Material}())
    type = lowercase(entity.type)

    if type == "diffuse"
        refl = pbrt_get_texture(entity, "reflectance", textures, (0.5, 0.5, 0.5))
        return Diffuse(Kd=refl)

    elseif type == "conductor"
        rough = pbrt_get_float_texture(entity, "roughness", textures, 0.0)
        # pbrt's `uroughness`/`vroughness` default to `roughness`; that fold is
        # only meaningful when `roughness` itself is a constant.
        combined_rough = if is_const_float(rough)
            rv = Float64(const_float(rough))
            urough = Float32(pbrt_get_float(entity, "uroughness", rv))
            vrough = Float32(pbrt_get_float(entity, "vroughness", rv))
            TexHandle(max(urough, vrough))
        else
            rough
        end
        remap = pbrt_get_bool(entity, "remaproughness", true)
        # Check for named spectra (gold, silver, copper, etc.)
        eta_str = pbrt_get_string(entity, "eta", "")
        k_str = pbrt_get_string(entity, "k", "")
        has_eta = !isempty(eta_str) || haskey(entity.params, "eta")
        has_k = !isempty(k_str) || haskey(entity.params, "k")
        has_refl = haskey(entity.params, "reflectance")
        if contains(eta_str, "Au") || contains(eta_str, "gold")
            return Gold(roughness=combined_rough, remap_roughness=remap)
        elseif contains(eta_str, "Ag") || contains(eta_str, "silver")
            return Silver(roughness=combined_rough, remap_roughness=remap)
        elseif contains(eta_str, "Cu") && !contains(eta_str, "CuZn")
            return Copper(roughness=combined_rough, remap_roughness=remap)
        elseif contains(eta_str, "Al")
            return Aluminum(roughness=combined_rough, remap_roughness=remap)
        end
        if has_refl
            refl = pbrt_get_rgb(entity, "reflectance", (1.0, 1.0, 1.0))
            return Conductor(reflectance=refl, roughness=combined_rough,
                             remap_roughness=remap)
        end
        if !has_eta && !has_k
            return Copper(roughness=combined_rough, remap_roughness=remap)
        end
        eta_rgb = pbrt_get_rgb(entity, "eta", (0.2, 0.2, 0.2))
        k_rgb = pbrt_get_rgb(entity, "k", (3.9, 3.9, 3.9))
        return Conductor(eta=eta_rgb, k=k_rgb, roughness=combined_rough,
                         remap_roughness=remap)

    elseif type == "dielectric"
        eta = Float32(pbrt_get_float(entity, "eta", 1.5))
        rough_base = pbrt_get_float_texture(entity, "roughness", textures, 0.0)
        rough_scalar = const_float(rough_base, 0f0)
        urough = pbrt_get_float_texture(entity, "uroughness", textures, Float64(rough_scalar))
        vrough = pbrt_get_float_texture(entity, "vroughness", textures, Float64(rough_scalar))
        remap = pbrt_get_bool(entity, "remaproughness", true)
        return Dielectric(index=eta, roughness=(urough, vrough), remap_roughness=remap)

    elseif type == "thindielectric"
        eta = Float32(pbrt_get_float(entity, "eta", 1.5))
        return ThinDielectric(eta=eta)

    elseif type == "coateddiffuse"
        refl = pbrt_get_texture(entity, "reflectance", textures, (0.5, 0.5, 0.5))
        # `roughness` may be a scalar OR a texture (Crown's `border_bott_inlay`
        # uses a mask texture). Silent coercion to a constant produced uniform
        # mirror-smooth coating where pbrt rendered a masked rough/smooth mix.
        rough = pbrt_get_float_texture(entity, "roughness", textures, 0.0)
        eta = Float32(pbrt_get_float(entity, "eta", 1.5))
        remap = pbrt_get_bool(entity, "remaproughness", true)
        return CoatedDiffuse(reflectance=refl, roughness=rough, eta=eta,
                             remap_roughness=remap)

    elseif type == "coatedconductor"
        irough = Float32(pbrt_get_float(entity, "interface.roughness", 0.0))
        crough = Float32(pbrt_get_float(entity, "conductor.roughness", 0.0))
        ieta = Float32(pbrt_get_float(entity, "interface.eta", 1.5))
        ceta_str = pbrt_get_string(entity, "conductor.eta", "")
        has_ceta = !isempty(ceta_str) || haskey(entity.params, "conductor.eta")
        has_ck = haskey(entity.params, "conductor.k")
        has_refl = haskey(entity.params, "reflectance")
        # Named spectra → pass PiecewiseLinearSpectrum directly as conductor eta/k
        # These get evaluated spectrally at render time via eval_ior_spectral
        if contains(ceta_str, "Au") || contains(ceta_str, "gold")
            return CoatedConductor(
                conductor_eta=AU_ETA_SPECTRUM, conductor_k=AU_K_SPECTRUM,
                conductor_roughness=crough,
                interface_roughness=irough, interface_eta=ieta)
        elseif contains(ceta_str, "Ag") || contains(ceta_str, "silver")
            return CoatedConductor(
                conductor_eta=AG_ETA_SPECTRUM, conductor_k=AG_K_SPECTRUM,
                conductor_roughness=crough,
                interface_roughness=irough, interface_eta=ieta)
        elseif contains(ceta_str, "Cu") && !contains(ceta_str, "CuZn")
            return CoatedConductor(
                conductor_eta=CU_ETA_SPECTRUM, conductor_k=CU_K_SPECTRUM,
                conductor_roughness=crough,
                interface_roughness=irough, interface_eta=ieta)
        elseif contains(ceta_str, "Al")
            return CoatedConductor(
                conductor_eta=AL_ETA_SPECTRUM, conductor_k=AL_K_SPECTRUM,
                conductor_roughness=crough,
                interface_roughness=irough, interface_eta=ieta)
        end
        if has_refl
            refl = pbrt_get_rgb(entity, "reflectance", (1.0, 1.0, 1.0))
            return CoatedConductor(reflectance=refl,
                                   conductor_roughness=crough,
                                   interface_roughness=irough,
                                   interface_eta=ieta)
        end
        if !has_ceta && !has_ck
            # pbrt-v4 default: Copper (metal-Cu-eta, metal-Cu-k)
            return CoatedConductor(
                conductor_eta=CU_ETA_SPECTRUM, conductor_k=CU_K_SPECTRUM,
                conductor_roughness=crough,
                interface_roughness=irough, interface_eta=ieta)
        end
        # Has explicit conductor.eta/k as RGB values
        ceta_rgb = pbrt_get_rgb(entity, "conductor.eta", (0.2, 0.2, 0.2))
        ck_rgb = pbrt_get_rgb(entity, "conductor.k", (3.9, 3.9, 3.9))
        return CoatedConductor(conductor_eta=ceta_rgb, conductor_k=ck_rgb,
                               conductor_roughness=crough,
                               interface_roughness=irough,
                               interface_eta=ieta)

    elseif type == "diffusetransmission"
        refl = pbrt_get_rgb(entity, "reflectance", (0.25, 0.25, 0.25))
        trans = pbrt_get_rgb(entity, "transmittance", (0.25, 0.25, 0.25))
        sc = Float32(pbrt_get_float(entity, "scale", 1.0))
        return DiffuseTransmission(reflectance=refl, transmittance=trans, scale=sc)

    elseif type == "mirror"
        refl = pbrt_get_rgb(entity, "reflectance", (0.9, 0.9, 0.9))
        return Mirror(Kr=refl)

    elseif type == "interface"
        # pbrt `Material "interface"` = nullptr surface: rays pass through, only
        # the medium swap fires. Wrapping in a transmissive Dielectric is
        # wrong: it makes shadow rays treat the boundary as opaque (intersection.jl:351)
        # and paints the volume's bounding mesh as a uniformly-shadowed cuboid.
        return NullMaterial()

    elseif type == "mix"
        haskey(entity.params, "materials") || error("mix material without 'materials' param")
        mat_names = String.(entity.params["materials"].values)
        length(mat_names) == 2 || error("mix material needs exactly 2 sub-materials, got $(length(mat_names))")
        # `amount` may be a scalar OR a texture (Crown's `bl_clamp_a` mixes a
        # coateddiffuse with a conductor using `clamp_mask.png`). Silent
        # coercion to 0.5 turned masked surfaces into a flat 50/50 blend.
        amount = pbrt_get_float_texture(entity, "amount", textures, 0.5)
        mat1 = get(mat_cache, mat_names[1], nothing)
        mat2 = get(mat_cache, mat_names[2], nothing)
        mat1 === nothing && error("mix: unknown material '$(mat_names[1])'")
        mat2 === nothing && error("mix: unknown material '$(mat_names[2])'")
        return MixMaterial(materials=(mat1, mat2), amount=amount)

    else
        @warn "pbrt: unsupported material type '$type', using default Diffuse"
        return Diffuse(Kd=(0.5, 0.5, 0.5))
    end
end

function resolve_pbrt_material(srec::PBRTShapeRecord, mat_cache::Dict{String, Material},
                               pbrt::PBRTScene; textures::AbstractDict{String}=Dict{String,TexHandle}())
    entity = if srec.material_name !== nothing
        name = srec.material_name
        if haskey(mat_cache, name)
            return mat_cache[name]
        elseif haskey(pbrt.named_materials, name)
            pbrt.named_materials[name]
        else
            @warn "pbrt: unknown named material '$name', using default"
            return Diffuse(Kd=(0.5, 0.5, 0.5))
        end
    elseif srec.material_inline !== nothing
        srec.material_inline
    else
        return Diffuse(Kd=(0.5, 0.5, 0.5))
    end

    mat = build_pbrt_material(entity, pbrt, textures, mat_cache)
    if mat !== nothing
        return attach_bump(mat, entity, textures)
    end
    return Diffuse(Kd=(0.5, 0.5, 0.5))
end

# ============================================================================
# Loop subdivision (pbrt "loopsubdiv" shape)
# ============================================================================

loop_beta(valence::Int) = valence == 3 ? 3f0 / 16f0 : 3f0 / (8f0 * valence)

function _loopsubdiv_adjacency(n::Int, faces::Vector{NTuple{3,Int}})
    neighbors = [Set{Int}() for _ in 1:n]
    for (v1, v2, v3) in faces
        push!(neighbors[v1], v2, v3)
        push!(neighbors[v2], v1, v3)
        push!(neighbors[v3], v1, v2)
    end
    return neighbors
end

function _loopsubdiv_boundary(n::Int, faces::Vector{NTuple{3,Int}})
    edge_count = Dict{Tuple{Int,Int}, Int}()
    for (v1, v2, v3) in faces
        for (a, b) in ((v1,v2), (v2,v3), (v3,v1))
            edge = minmax(a, b)
            edge_count[edge] = get(edge_count, edge, 0) + 1
        end
    end
    boundary = falses(n)
    for ((a, b), count) in edge_count
        count == 1 && (boundary[a] = true; boundary[b] = true)
    end
    return boundary
end

function _loopsubdiv_once(vertices::Vector{Point3f}, faces::Vector{NTuple{3,Int}})
    neighbors = _loopsubdiv_adjacency(length(vertices), faces)
    boundary = _loopsubdiv_boundary(length(vertices), faces)

    new_vertices = Vector{Point3f}(undef, length(vertices))
    for (i, v) in enumerate(vertices)
        neighs = collect(neighbors[i])
        valence = length(neighs)
        if boundary[i]
            boundary_neighs = filter(n -> boundary[n], neighs)
            if length(boundary_neighs) >= 2
                bn = boundary_neighs[1:2]
                new_vertices[i] = 0.75f0 * v + 0.125f0 * (vertices[bn[1]] + vertices[bn[2]])
            else
                new_vertices[i] = v
            end
        else
            b = loop_beta(valence)
            ring_sum = sum(vertices[n] for n in neighs)
            new_vertices[i] = (1f0 - valence * b) * v + b * ring_sum
        end
    end

    edge_faces = Dict{Tuple{Int,Int}, Vector{Int}}()
    for (fi, (v1, v2, v3)) in enumerate(faces)
        for (a, b) in ((v1,v2), (v2,v3), (v3,v1))
            edge = minmax(a, b)
            push!(get!(edge_faces, edge, Int[]), fi)
        end
    end

    edge_vertex_map = Dict{Tuple{Int,Int}, Int}()
    out_vertices = copy(new_vertices)
    for (edge, adj_faces) in edge_faces
        a, b = edge
        new_p = if length(adj_faces) == 1
            0.5f0 * (vertices[a] + vertices[b])
        else
            f1, f2 = adj_faces[1], adj_faces[2]
            opp1 = only(filter(v -> v != a && v != b, (faces[f1][1], faces[f1][2], faces[f1][3])))
            opp2 = only(filter(v -> v != a && v != b, (faces[f2][1], faces[f2][2], faces[f2][3])))
            0.375f0 * (vertices[a] + vertices[b]) + 0.125f0 * (vertices[opp1] + vertices[opp2])
        end
        push!(out_vertices, new_p)
        edge_vertex_map[edge] = length(out_vertices)
    end

    new_faces = NTuple{3,Int}[]
    sizehint!(new_faces, 4 * length(faces))
    for (v1, v2, v3) in faces
        e12 = edge_vertex_map[minmax(v1, v2)]
        e23 = edge_vertex_map[minmax(v2, v3)]
        e31 = edge_vertex_map[minmax(v3, v1)]
        push!(new_faces, (v1, e12, e31), (v2, e23, e12), (v3, e31, e23), (e12, e23, e31))
    end

    return out_vertices, new_faces
end

function loop_subdivide(vertices::Vector{Point3f}, faces::Vector{NTuple{3,Int}}, levels::Int)
    v, f = vertices, faces
    for _ in 1:levels
        v, f = _loopsubdiv_once(v, f)
    end
    return v, f
end

# ============================================================================
# Shape building
# ============================================================================

function build_pbrt_shape(srec::PBRTShapeRecord, pbrt::PBRTScene)
    entity = srec.entity
    type = lowercase(entity.type)

    if type == "sphere"
        radius = Float32(pbrt_get_float(entity, "radius", 1.0))
        # pbrt-v4 spheres support partial extent via zmin/zmax (in object space)
        # and phimax (azimuth). The Crown's dome and shadow_*gold test scenes
        # carve a half-sphere bowl with `"float zmin" -radius "float zmax" 0`;
        # ignoring these silently produced a phantom mirror half-sphere
        # intersecting the receiver plane.
        zmin = Float32(pbrt_get_float(entity, "zmin", -radius))
        zmax = Float32(pbrt_get_float(entity, "zmax",  radius))
        phimax = Float32(pbrt_get_float(entity, "phimax", 360.0))
        # pbrt's sphere is analytic; we tessellate. 64 segments is too coarse
        # for smooth-mirror conductors (visible facet aliasing on the
        # silhouette and on the reflection's bright bowl). 512 segments brings
        # the silhouette to <0.2° per vertex, sub-pixel on 256×256 references,
        # and gets the smooth-mirror reflection's per-pixel value within a
        # few percent of pbrt's analytic-sphere reference across the test
        # suite. Higher counts have diminishing returns vs BLAS build cost.
        mesh = tessellate_sphere(radius; segments=512, zmin=zmin, zmax=zmax, phimax_deg=phimax)
        mesh = apply_pbrt_transform(mesh, srec.transform)

    elseif type == "disk"
        radius = Float32(pbrt_get_float(entity, "radius", 1.0))
        mesh = tessellate_disk(radius; segments=64)
        mesh = apply_pbrt_transform(mesh, srec.transform)

    elseif type == "trianglemesh"
        mesh = build_trianglemesh(entity, srec.transform)

    elseif type == "plymesh"
        filename = pbrt_get_string(entity, "filename", "")
        isempty(filename) && (@warn "pbrt: plymesh without filename"; return nothing)
        path = isabspath(filename) ? filename : joinpath(pbrt.base_dir, filename)
        isfile(path) || (@warn "pbrt: PLY file not found: $path"; return nothing)
        mesh = FileIO.load(path)
        mesh = apply_pbrt_transform(mesh, srec.transform)

    elseif type == "loopsubdiv"
        levels = pbrt_get_int(entity, "levels", 3)
        base_mesh = build_trianglemesh(entity, srec.transform)
        if base_mesh === nothing
            return nothing
        end
        pts = collect(Point3f, GeometryBasics.coordinates(base_mesh))
        fs = [NTuple{3,Int}((f[1], f[2], f[3])) for f in GeometryBasics.faces(base_mesh)]
        sub_pts, sub_fs = loop_subdivide(pts, fs, levels)
        fi = [TriangleFace{Int}(f[1], f[2], f[3]) for f in sub_fs]
        mesh = GeometryBasics.normal_mesh(GeometryBasics.Mesh(sub_pts, fi))

    else
        @warn "pbrt: unsupported shape type '$type'"
        return nothing
    end

    # ReverseOrientation flips face winding, which flips computed normals.
    # This is critical for one-sided area lights that only emit from the normal side.
    if mesh !== nothing && srec.reverse_orientation
        mesh = reverse_mesh_orientation(mesh)
    end

    return mesh
end

function reverse_mesh_orientation(mesh)
    positions = collect(GeometryBasics.coordinates(mesh))
    faces = collect(GeometryBasics.faces(mesh))
    flipped = [TriangleFace{Int}(f[1], f[3], f[2]) for f in faces]
    return GeometryBasics.Mesh(positions, flipped)
end

function build_trianglemesh(entity::PBRTEntity, transform::Mat4f)
    P = pbrt_get_floats(entity, "P")
    indices = pbrt_get_ints(entity, "indices")
    (isempty(P) || isempty(indices)) && return nothing

    n_verts = length(P) ÷ 3
    points = [Point3f(P[3i-2], P[3i-1], P[3i]) for i in 1:n_verts]

    # Apply transform
    if transform != IDENTITY4
        for i in eachindex(points)
            p = points[i]
            p4 = transform * Vec4f(p[1], p[2], p[3], 1f0)
            points[i] = Point3f(p4[1] / p4[4], p4[2] / p4[4], p4[3] / p4[4])
        end
    end

    n_tris = length(indices) ÷ 3
    # pbrt uses 0-based indices
    faces = [TriangleFace{Int}(indices[3i-2]+1, indices[3i-1]+1, indices[3i]+1)
             for i in 1:n_tris]

    # Optional normals and UVs
    N = pbrt_get_floats(entity, "N")
    uv = pbrt_get_floats(entity, "uv")
    has_normals = !isempty(N) && length(N) == 3 * n_verts
    has_uvs = !isempty(uv) && length(uv) == 2 * n_verts

    kwargs = Dict{Symbol, Any}()
    if has_normals
        kwargs[:normal] = [Normal3f(N[3i-2], N[3i-1], N[3i]) for i in 1:n_verts]
    end
    if has_uvs
        kwargs[:uv] = [Point2f(uv[2i-1], uv[2i]) for i in 1:n_verts]
    end

    return GeometryBasics.Mesh(points, faces; kwargs...)
end

function apply_pbrt_transform(mesh, transform::Mat4f)
    transform == IDENTITY4 && return mesh

    positions = GeometryBasics.coordinates(mesh)
    faces = GeometryBasics.faces(mesh)
    new_positions = map(positions) do p
        p4 = transform * Vec4f(p[1], p[2], p[3], 1f0)
        Point3f(p4[1] / p4[4], p4[2] / p4[4], p4[3] / p4[4])
    end
    # Preserve per-vertex normals and UVs through the rebuild. Normals
    # transform by the upper-3x3 (assuming no non-uniform scale on a unit
    # surface — pbrt's Transform stack is rotation+translation+uniform-scale
    # for our test scenes), then renormalize.
    props = propertynames(mesh)
    has_n = :normal in props
    has_uv = :uv in props
    if !(has_n || has_uv)
        return GeometryBasics.Mesh(collect(new_positions), collect(faces))
    end
    R = SMatrix{3,3,Float32}(
        transform[1,1], transform[2,1], transform[3,1],
        transform[1,2], transform[2,2], transform[3,2],
        transform[1,3], transform[2,3], transform[3,3])
    new_normals = if has_n
        ns = mesh.normal
        nout = Vector{Vec3f}(undef, length(ns))
        for i in eachindex(ns)
            n = R * Vec3f(ns[i][1], ns[i][2], ns[i][3])
            l = sqrt(dot(n, n))
            nout[i] = l > 0f0 ? n / l : n
        end
        nout
    else
        nothing
    end
    new_uvs = has_uv ? collect(mesh.uv) : nothing
    if has_n && has_uv
        return GeometryBasics.Mesh(collect(new_positions), collect(faces);
                                   normal=new_normals, uv=new_uvs)
    elseif has_n
        return GeometryBasics.Mesh(collect(new_positions), collect(faces);
                                   normal=new_normals)
    else
        return GeometryBasics.Mesh(collect(new_positions), collect(faces);
                                   uv=new_uvs)
    end
end

# UV sphere tessellation
function tessellate_sphere(radius::Float32; segments::Int=64,
                           zmin::Float32=-radius, zmax::Float32=radius,
                           phimax_deg::Float32=360f0)
    # pbrt-v4 sphere: z ∈ [zmin, zmax], azimuth φ ∈ [0, phimax].
    # Theta from the +z axis: theta_min ↔ zmax (cosθ = zmax/r),
    #                         theta_max ↔ zmin (cosθ = zmin/r).
    # We emit per-vertex normals (the unit radial direction) and UVs
    # (u=φ/phimax, v=(θ-θmin)/(θmax-θmin)) so smooth shading and bump-mapped
    # spheres match pbrt's analytic sphere.
    zmin = clamp(zmin, -radius,  radius)
    zmax = clamp(zmax, -radius,  radius)
    @assert zmax >= zmin "sphere: zmax must be >= zmin"
    theta_min = acos(clamp(zmax / radius, -1f0, 1f0))
    theta_max = acos(clamp(zmin / radius, -1f0, 1f0))
    phimax = clamp(deg2rad(phimax_deg), 0f0, 2f0 * Float32(π))
    full_phi = phimax >= 2f0 * Float32(π) - 1f-5
    has_top    = theta_min < 1f-5                       # includes north pole
    has_bottom = theta_max > Float32(π) - 1f-5          # includes south pole

    n_phi = full_phi ? segments : max(2, Int(round(segments * phimax / (2f0 * Float32(π)))))
    n_verts_per_ring = full_phi ? segments : n_phi + 1
    rings = max(2, segments ÷ 2)
    inv_radius = 1f0 / radius
    phi_step_full = 2f0 * Float32(π) / segments

    points  = Point3f[]
    normals = Vec3f[]
    uvs     = Point2f[]
    faces   = TriangleFace{Int}[]

    function push_ring_vert!(theta::Float32, j::Int)
        st, ct = sincos(theta)
        phi = full_phi ? phi_step_full * (j - 1) :
              phimax * (j - 1) / (n_verts_per_ring - 1)
        sp, cp = sincos(phi)
        x = radius * st * cp; y = radius * st * sp; z = radius * ct
        push!(points,  Point3f(x, y, z))
        push!(normals, Vec3f(x * inv_radius, y * inv_radius, z * inv_radius))
        u = full_phi ? phi_step_full * (j - 1) / phimax :
                       (j - 1) / (n_verts_per_ring - 1)
        # v runs from the LOW-z end to the high-z end, which is the opposite of
        # what the local `theta_min`/`theta_max` names suggest. pbrt's
        # `thetaZMin` is `acos(min(zMin, zMax) / radius)` (shapes.h:126) and
        # `acos` is decreasing, so its `thetaZMin` is the LARGER theta — the
        # bottom of the sphere — and `v = (theta - thetaZMin)/(thetaZMax -
        # thetaZMin)` (shapes.h:245) is 0 there, rising to 1 at the top.
        # Reading those names geometrically instead gave `v = 1 - pbrt_v` on
        # every sphere in the scene set, mirroring every texture and bump map
        # about the equator.
        v = (theta_max - theta) / max(theta_max - theta_min, 1f-8)
        push!(uvs, Point2f(u, v))
    end

    if has_top
        push!(points,  Point3f(0f0, 0f0,  radius))
        push!(normals, Vec3f(0f0, 0f0,  1f0))
        push!(uvs,     Point2f(0f0, 1f0))   # +z pole is v=1 in pbrt's parameterization
    end
    top_idx    = has_top ? 1 : 0
    first_ring = has_top ? 2 : 1

    first_body_ring = has_top ? 1 : 0
    last_body_ring  = has_bottom ? rings - 1 : rings
    for i in first_body_ring:last_body_ring
        theta = theta_min + (theta_max - theta_min) * i / rings
        for j in 1:n_verts_per_ring
            push_ring_vert!(theta, j)
        end
    end
    n_body_rings = last_body_ring - first_body_ring + 1

    if has_bottom
        push!(points,  Point3f(0f0, 0f0, -radius))
        push!(normals, Vec3f(0f0, 0f0, -1f0))
        push!(uvs,     Point2f(0f0, 0f0))   # -z pole is v=0
    end
    bottom_idx = has_bottom ? length(points) : 0

    # Winding: vertices listed CCW when viewed from OUTSIDE the sphere so
    # `cross(b-a, c-a)` points along the same hemisphere as the radial
    # vertex normals (Raycore's geometric-normal convention).
    if has_top
        for j in 1:n_verts_per_ring
            j_next = full_phi ? mod1(j + 1, segments) : j + 1
            j_next > n_verts_per_ring && continue
            push!(faces, TriangleFace{Int}(top_idx,
                                           first_ring + j - 1,
                                           first_ring + j_next - 1))
        end
    end

    for i in 0:(n_body_rings - 2)
        for j in 1:n_verts_per_ring
            j_next = full_phi ? mod1(j + 1, segments) : j + 1
            j_next > n_verts_per_ring && continue
            a = first_ring + i * n_verts_per_ring + (j - 1)
            b = first_ring + i * n_verts_per_ring + (j_next - 1)
            c = first_ring + (i + 1) * n_verts_per_ring + (j - 1)
            d = first_ring + (i + 1) * n_verts_per_ring + (j_next - 1)
            push!(faces, TriangleFace{Int}(a, c, b))
            push!(faces, TriangleFace{Int}(b, c, d))
        end
    end

    if has_bottom
        last_ring_start = first_ring + (n_body_rings - 1) * n_verts_per_ring
        for j in 1:n_verts_per_ring
            j_next = full_phi ? mod1(j + 1, segments) : j + 1
            j_next > n_verts_per_ring && continue
            push!(faces, TriangleFace{Int}(bottom_idx,
                                           last_ring_start + j_next - 1,
                                           last_ring_start + j - 1))
        end
    end

    return GeometryBasics.Mesh(points, faces; normal=normals, uv=uvs)
end

function tessellate_disk(radius::Float32; segments::Int=64)
    points = Point3f[Point3f(0f0, 0f0, 0f0)]
    faces = TriangleFace{Int}[]

    for j in 1:segments
        phi = 2f0 * Float32(π) * (j - 1) / segments
        sp, cp = sincos(phi)
        push!(points, Point3f(radius * cp, radius * sp, 0f0))
    end

    for j in 1:segments
        j_next = mod1(j + 1, segments)
        push!(faces, TriangleFace{Int}(1, 1 + j, 1 + j_next))
    end

    return GeometryBasics.Mesh(points, faces)
end

# ============================================================================
# Light building
# ============================================================================

function build_pbrt_light(lrec::PBRTLightRecord, pbrt::PBRTScene)
    entity = lrec.entity
    type = lowercase(entity.type)

    if type == "point"
        rgb = pbrt_get_rgb(entity, "I", (1.0, 1.0, 1.0))
        sc = Float32(pbrt_get_float(entity, "scale", 1.0))
        from = pbrt_get_rgb(entity, "from", (0.0, 0.0, 0.0))
        pos = Vec3f(Float32(from[1]), Float32(from[2]), Float32(from[3]))
        if lrec.transform != IDENTITY4
            p4 = lrec.transform * Vec4f(pos[1], pos[2], pos[3], 1f0)
            pos = Vec3f(p4[1] / p4[4], p4[2] / p4[4], p4[3] / p4[4])
        end
        # Use RGB constructor (not RGBSpectrum) — matches Hikari's expected intensity model
        return PointLight(
            RGB{Float32}(Float32(rgb[1]) * sc, Float32(rgb[2]) * sc, Float32(rgb[3]) * sc),
            pos)

    elseif type == "distant"
        rgb = pbrt_get_rgb(entity, "L", (1.0, 1.0, 1.0))
        sc = Float32(pbrt_get_float(entity, "scale", 1.0))
        from = pbrt_get_rgb(entity, "from", (0.0, 0.0, 0.0))
        to = pbrt_get_rgb(entity, "to", (0.0, 0.0, 1.0))
        dir = Vec3f(Float32(to[1] - from[1]), Float32(to[2] - from[2]),
                    Float32(to[3] - from[3]))
        return DirectionalLight(
            RGB{Float32}(Float32(rgb[1]) * sc, Float32(rgb[2]) * sc, Float32(rgb[3]) * sc),
            dir)

    elseif type == "spot"
        rgb = pbrt_get_rgb(entity, "I", (1.0, 1.0, 1.0))
        sc = Float32(pbrt_get_float(entity, "scale", 1.0))
        cone = Float32(pbrt_get_float(entity, "coneangle", 30.0))
        delta = Float32(pbrt_get_float(entity, "conedeltaangle", 5.0))
        from = pbrt_get_rgb(entity, "from", (0.0, 0.0, 0.0))
        pos = Point3f(Float32(from[1]), Float32(from[2]), Float32(from[3]))
        to = pbrt_get_rgb(entity, "to", (0.0, 0.0, 1.0))
        target = Point3f(Float32(to[1]), Float32(to[2]), Float32(to[3]))
        # SpotLight(rgb, position, target, total_width_deg, falloff_start_deg)
        return SpotLight(
            RGB{Float32}(Float32(rgb[1]) * sc, Float32(rgb[2]) * sc, Float32(rgb[3]) * sc),
            pos, target, cone, cone - delta)

    elseif type == "infinite"
        filename = pbrt_get_string(entity, "filename", "")
        sc = Float32(pbrt_get_float(entity, "scale", 1.0))
        if !isempty(filename)
            path = isabspath(filename) ? filename : joinpath(pbrt.base_dir, filename)
            if isfile(path)
                # Extract 3x3 rotation from the light's 4x4 transform
                t = lrec.transform
                rotation = Mat3f(t[1,1], t[2,1], t[3,1],
                                 t[1,2], t[2,2], t[3,2],
                                 t[1,3], t[2,3], t[3,3])
                # pbrt normalizes: scale /= SpectrumToPhotometric(colorSpace.illuminant)
                sc /= D65_PHOTOMETRIC
                # Convert non-sRGB images to sRGB (Hikari's spectral uplift uses sRGB tables)
                env_path_converted = convert_envmap_to_srgb(path)
                return EnvironmentLight(env_path_converted; scale=RGBSpectrum(sc), rotation=rotation)
            end
        end
        rgb = pbrt_get_rgb(entity, "L", (1.0, 1.0, 1.0))
        return AmbientLight(
            RGB{Float32}(Float32(rgb[1]) * sc, Float32(rgb[2]) * sc, Float32(rgb[3]) * sc))

    else
        @warn "pbrt: unsupported light type '$type'"
        return nothing
    end
end

# ACES AP0 → sRGB conversion matrix (includes XYZ intermediate)
const SRGB_FROM_ACES_AP0 = Mat3f(
     2.55798f0,  -0.27799f0,  -0.01717f0,
    -1.11929f0,   1.36605f0,  -0.14857f0,
    -0.39175f0,  -0.09349f0,   1.08128f0,
)

"""
Detect if an EXR image is in ACES color space and convert to sRGB if needed.
Returns the path to use (original if already sRGB, or a temp converted file).
"""
function convert_envmap_to_srgb(path::String)
    # Try to detect ACES from file content. pbrt's imgtool reports "color space: ACES"
    # for ACES images. We detect by checking if the EXR has chromaticities matching ACES AP0.
    # For simplicity: check if pbrt's imgtool reports ACES, or use a heuristic based on
    # the file path or a quick pixel range check.
    #
    # Pragmatic approach: try to read chromaticities from EXR metadata.
    # If not available, check pixel values — ACES images often have values > 1 for HDR
    # but that's not definitive. Instead, use a simple approach:
    # run pbrt's imgtool if available, or just convert all HDR env maps defensively.

    # Check via imgtool
    pbrt_imgtool = "/sim/Programmieren/VulkanDev/pbrt-v4/build/imgtool"
    is_aces = false
    if isfile(pbrt_imgtool)
        # A failure here is not "not ACES" — it is "we do not know", and the
        # consequence is an ACES image used as if it were sRGB, which is a
        # wrong picture rather than an error. So it says so instead of
        # pretending the answer was no.
        try
            output = read(`$pbrt_imgtool info $path`, String)
            is_aces = contains(output, "ACES")
        catch e
            @warn "could not ask imgtool about $path; treating it as sRGB" exception = (e, catch_backtrace())
        end
    end

    if !is_aces
        return path  # sRGB or unknown — use as-is
    end

    # Convert ACES → sRGB
    img = FileIO.load(path)
    M = SRGB_FROM_ACES_AP0
    converted = map(img) do p
        r, g, b = Float32(p.r), Float32(p.g), Float32(p.b)
        nr = M[1,1]*r + M[1,2]*g + M[1,3]*b
        ng = M[2,1]*r + M[2,2]*g + M[2,3]*b
        nb = M[3,1]*r + M[3,2]*g + M[3,3]*b
        RGB{Float32}(max(0f0, nr), max(0f0, ng), max(0f0, nb))
    end
    # Save to temp file
    converted_path = tempname() * ".exr"
    FileIO.save(converted_path, converted)
    @info "pbrt: converted ACES env map to sRGB" path converted_path
    return converted_path
end

# ============================================================================
# Convenience render function
# ============================================================================

"""
    render_pbrt(filename; backend, samples, max_depth, hw_accel, output)

Load a pbrt scene, render it, and optionally save to file. Returns the rendered image.
"""
function render_pbrt(filename::AbstractString;
                     backend=KA.CPU(),
                     samples::Union{Nothing, Int}=nothing,
                     max_depth::Union{Nothing, Int}=nothing,
                     hw_accel::Bool=false,
                     output::Union{Nothing, String}=nothing)
    r = load_pbrt(filename; backend=backend, samples=samples, max_depth=max_depth, hw_accel=hw_accel)
    s = r.integrator_settings
    spp = s.samples
    vp = VolPath(samples=spp, max_depth=s.max_depth, regularize=s.regularize,
                 russian_roulette_depth=s.russian_roulette_depth,
                 max_component_value=s.max_component_value,
                 sensor=r.sensor, hw_accel=hw_accel)

    img = vp(r.scene, r.film, r.camera)

    if output !== nothing
        FileIO.save(output, Array(img))
    end
    return r.film.framebuffer
end

"""
    apply_sensor!(framebuffer, sensor)

Apply pixel sensor ISO/exposure and white balance correction to a rendered framebuffer.
Hikari renders using CIE XYZ → sRGB. This function applies:
1. imaging_ratio (ISO * exposure_time / 100) — brightness scaling
2. White balance — chromatic adaptation from scene illuminant to D65

Note: Real camera sensor spectral response curves (e.g., nikon_d850) change the
spectral integration itself and cannot be applied as a post-process. They require
modification to Hikari's spectral→XYZ conversion kernel. White balance and ISO
are correctly handled here as they are linear transforms on the XYZ/RGB values.
"""
function apply_sensor!(fb::AbstractMatrix{RGB{Float32}}, sensor::PixelSensor)
    # Compute correction: sRGB → XYZ → (sensor output_from_sensor) = sRGB → sensor_sRGB
    # For cie1931 with WB: output_from_sensor = SRGB_FROM_XYZ * white_balance_matrix
    # So correction = (SRGB_FROM_XYZ * WB) * XYZ_FROM_SRGB = SRGB * WB * inv(SRGB)
    # For cie1931 without WB: output_from_sensor = SRGB_FROM_XYZ → correction = identity
    correction = sensor.output_from_sensor * XYZ_FROM_SRGB
    ratio = sensor.imaging_ratio

    # Skip if it's the default pipeline (identity correction, ratio=1)
    if correction ≈ Mat3f(LinearAlgebra.I) && ratio ≈ 1f0
        return
    end

    M = correction * ratio

    for idx in eachindex(fb)
        p = fb[idx]
        v = Vec3f(p.r, p.g, p.b)
        v2 = M * v
        fb[idx] = RGB{Float32}(max(0f0, v2[1]), max(0f0, v2[2]), max(0f0, v2[3]))
    end
end
