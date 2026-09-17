#
# pbrt reference suite — shared helpers
#
# Single source of truth for:
#   * locating scenes + references
#   * rendering a scene with Hikari (VolPath), parametrized by backend / spp /
#     max_depth / hw_accel
#   * loading a pre-committed pbrt-v4 EXR reference (optionally generating it
#     if the pbrt binary is available)
#   * image comparison metrics (log-space tile p95, energy ratio, mean/median
#     per-pixel diff)
#
# This file defines functions only — it does not run tests. It is `include`d
# by test_pbrt_all_materials.jl (normal @testset runner) and by
# run_comparison.jl (gallery-generating script). Both drive the same
# render/compare code so they can't drift.

using Hikari
using Lava, Mantle
using FileIO
using Statistics
using Colors
using ImageFiltering
import KernelAbstractions as KA

const SCENES_DIR = joinpath(@__DIR__, "scenes")
const REFS_DIR   = joinpath(@__DIR__, "references")
const PBRT_BIN   = get(ENV, "PBRT_BIN", "/sim/Programmieren/VulkanDev/pbrt-v4/build/pbrt")

"""Is hardware ray tracing available on the currently-bound Lava device?"""
function hw_rt_available()
    try
        Mantle.supports_rt_pipeline(Mantle.defaultbackend())
    catch
        false
    end
end

# ── Reference management ────────────────────────────────────────────────────

"""
    ensure_reference(scene_name; spp) -> Matrix{RGB{Float32}}

Load the pbrt-v4 reference EXR for `scene_name`. If missing, fall back to
running the pbrt binary at `PBRT_BIN` to generate it. Throws if neither is
available.
"""
# Every committed reference is rendered at this sample count, and the suite
# verifies it rather than trusting it.
#
# It is deliberately a constant and not the caller's `samples`, because the
# references drift apart otherwise: of 167, eight sit at 64 spp, two at 512 and
# one at 1024 while Hikari renders all of them at 256. For the 64-spp eight that
# means
# pbrt carried 4x fewer samples, so the reference was roughly 2x noisier than
# the image under test — and since the tile score is a p95 of per-pixel
# difference, that noise was scored as Hikari error. Those eight average tile
# 0.0169 against 0.0097 across the whole set.
#
# Keying it to `samples` instead would be worse than the bug: a CI run at 32 spp
# would quietly rewrite all 167 committed references at 32.
const REFERENCE_SPP = 256

"""
    exr_samples_per_pixel(path) -> Int or nothing

The `samplesPerPixel` attribute pbrt writes into an EXR header. `nothing` when
the attribute is absent, which is how a reference from another source reads.
"""
function exr_samples_per_pixel(path::AbstractString)
    bytes = open(io -> read(io, 4096), path, "r")
    key = Vector{UInt8}(codeunits("samplesPerPixel"))
    idx = findfirst(key, bytes)
    idx === nothing && return nothing
    p = last(idx) + 2                                   # past the name's NUL
    tend = findnext(==(0x00), bytes, p)
    tend === nothing && return nothing
    String(bytes[p:tend-1]) == "int" || return nothing
    p = tend + 1 + 4                                    # past type NUL + size field
    return Int(only(reinterpret(Int32, bytes[p:p+3])))
end

"""Render `scene_name`'s pbrt-v4 reference at `REFERENCE_SPP`, overwriting."""
function generate_reference(scene_name::AbstractString)
    isfile(PBRT_BIN) || error("pbrt binary not found at $PBRT_BIN (set PBRT_BIN)")
    scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
    ref_file   = joinpath(REFS_DIR, "$(scene_name).exr")
    @info "Generating reference: $scene_name at $(REFERENCE_SPP) spp"
    # `--spp` overrides whatever `Sampler` the scene asks for, which matters:
    # six scenes pin `pixelsamples 64` in their own .pbrt.
    run(`$PBRT_BIN $scene_file --outfile $ref_file --spp $REFERENCE_SPP`)
    return nothing
end

function ensure_reference(scene_name::AbstractString)
    ref_file = joinpath(REFS_DIR, "$(scene_name).exr")
    isfile(ref_file) || generate_reference(scene_name)

    got = exr_samples_per_pixel(ref_file)
    if got !== nothing && got != REFERENCE_SPP
        error("""
              Reference for $scene_name was rendered at $(got) spp, not $(REFERENCE_SPP).

              Hikari renders at $(REFERENCE_SPP), so comparing against this comes
              with a different noise floor on each side and the difference is
              scored as Hikari error. Regenerate it with the same settings:

                  julia --project=. -e 'include("test/pbrt/suite.jl"); generate_reference("$scene_name")'
              """)
    end
    return FileIO.load(ref_file)
end

# ── Rendering ───────────────────────────────────────────────────────────────

"""
    render_scene(scene_name; backend, samples, max_depth, hw_accel, close_vp=true)
        -> Matrix{RGB{Float32}}

Render `scene_name` with Hikari's VolPath integrator and return a host-side
framebuffer. All knobs are explicit — no defaults baked into globals. When
`hw_accel=true` the scene is built with `VulkanTLAS` and VolPath dispatches to
the hardware RT path.
"""
function render_scene(scene_name::AbstractString;
                      backend=Mantle.defaultbackend(),
                      samples::Int=256,
                      max_depth::Union{Nothing,Int}=nothing,
                      hw_accel::Bool=false,
                      close_vp::Bool=true)
    scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
    isfile(scene_file) || error("pbrt scene not found: $scene_file")

    r = Hikari.load_pbrt(scene_file;
                         backend=backend,
                         samples=samples,
                         max_depth=max_depth,
                         hw_accel=hw_accel)
    s = r.integrator_settings
    vp = Hikari.VolPath(samples=samples,
                        max_depth=s.max_depth,
                        regularize=s.regularize,
                        russian_roulette_depth=s.russian_roulette_depth,
                        max_component_value=s.max_component_value,
                        sensor=r.sensor,
                        hw_accel=hw_accel)
    try
        vp(r.scene, r.film, r.camera)
        return Array(r.film.framebuffer)
    finally
        close_vp && close(vp)
    end
end

# ── Metrics ─────────────────────────────────────────────────────────────────

"""2×2 gaussian-filtered downsample. Reduces MC noise by ~2× so tile scores
are dominated by spatial bugs rather than per-pixel variance."""
function downsample_2x(img::AbstractMatrix)
    blurred = imfilter(img, Kernel.gaussian((0.75, 0.75)))
    return blurred[1:2:end, 1:2:end]
end

"""
    tile_score(a, b; tile_size=16, percentile=0.95) -> Float64

95th-percentile tile-based mean color distance in log1p space. Images are
downsampled 2× first. Returns `Inf` if sizes disagree.
"""
function tile_score(a::AbstractMatrix, b::AbstractMatrix;
                    tile_size::Int=16, percentile::Float64=0.95)
    size(a) == size(b) || return Inf
    a = downsample_2x(a); b = downsample_2x(b)
    h, w = size(a)
    rh = round.(Int, range(0, h, length=max(2, ceil(Int, h / tile_size))))
    rw = round.(Int, range(0, w, length=max(2, ceil(Int, w / tile_size))))
    bnd(r) = zip(r[1:end-1] .+ 1, r[2:end])
    function d(p1, p2)
        r1 = log1p(max(0.0, Float64(red(p1))));   g1 = log1p(max(0.0, Float64(green(p1))));   b1 = log1p(max(0.0, Float64(blue(p1))))
        r2 = log1p(max(0.0, Float64(red(p2))));   g2 = log1p(max(0.0, Float64(green(p2))));   b2 = log1p(max(0.0, Float64(blue(p2))))
        sqrt((r1-r2)^2+(g1-g2)^2+(b1-b2)^2)
    end
    scores = [mean(d.(@view(a[r1:r2,c1:c2]), @view(b[r1:r2,c1:c2])))
              for (r1,r2) in bnd(rh) for (c1,c2) in bnd(rw)]
    sort!(scores)
    return scores[clamp(ceil(Int, percentile * length(scores)), 1, length(scores))]
end

"""Total-luminance ratio: `sum(rec) / sum(ref)`. 1.0 = matched energy."""
function energy_ratio(ref::AbstractMatrix, rec::AbstractMatrix)
    total_ref = 0.0; total_rec = 0.0
    @inbounds for i in eachindex(ref, rec)
        rp = ref[i]; hp = rec[i]
        total_ref += Float64(red(rp)) + Float64(green(rp)) + Float64(blue(rp))
        total_rec += Float64(red(hp)) + Float64(green(hp)) + Float64(blue(hp))
    end
    return total_rec / max(total_ref, 1e-10)
end

"""All metrics in one pass."""
function compute_metrics(ref::AbstractMatrix, rec::AbstractMatrix)
    @assert size(ref) == size(rec) "size mismatch: $(size(ref)) vs $(size(rec))"
    diffs = Float32[]
    sizehint!(diffs, length(ref))
    @inbounds for i in eachindex(ref, rec)
        rp = ref[i]; hp = rec[i]
        rr, rg, rb = Float32(red(rp)), Float32(green(rp)), Float32(blue(rp))
        hr, hg, hb = Float32(red(hp)), Float32(green(hp)), Float32(blue(hp))
        push!(diffs, (abs(rr-hr) + abs(rg-hg) + abs(rb-hb)) / 3)
    end
    return (
        tile         = tile_score(ref, rec),
        energy_ratio = energy_ratio(ref, rec),
        mean_diff    = mean(diffs),
        median_diff  = median(diffs),
        max_diff     = maximum(diffs),
    )
end

"""Scene names present on disk under `SCENES_DIR` (without `.pbrt` suffix),
filtered by `prefix`. Sorted for determinism."""
function list_scenes(prefix::AbstractString="")
    files = filter(f -> startswith(f, prefix) && endswith(f, ".pbrt"), readdir(SCENES_DIR))
    return sort!([replace(f, ".pbrt" => "") for f in files])
end

# ── Pass/fail band ──────────────────────────────────────────────────────────
# Log-space tile p95 under 0.07, energy within ±5 %. Uniform for every scene
# class: no per-scene overrides and no SPP-dependent relaxation, because the
# suite renders at high spp (256 by default) so the Monte-Carlo noise floor sits
# well under the tile limit. A scene that cannot clear this band is a rendering
# bug to fix, not a threshold to widen.
#
# Defined once, here, because the gallery colours its cells by the same numbers.
# Written out in the runner and in the gallery separately the two drift: a
# gallery applying a looser band below 128 spp (tile
# 0.10, energy ±10 %) plus a per-scene override table reaching tile=0.55 for
# scenes the runner checks at 0.07, so a scene could read green there and fail
# the suite.
const TILE_THRESHOLD = 0.07
const ENERGY_LOW     = 0.95
const ENERGY_HIGH    = 1.05

"""Does a scene's tile/energy pair land inside the band above?"""
within_tolerance(tile::Real, energy::Real) =
    tile < TILE_THRESHOLD && ENERGY_LOW < energy < ENERGY_HIGH

const RECORD_DIR = joinpath(@__DIR__, "recorded")

# SW keeps the flat `recorded/<name>.exr` layout that `comparison_app.jl` and
# `export_static.jl` already read; HW goes in a subdirectory so a run of both
# backends does not have one overwrite the other.
record_dir(hw_accel::Bool)    = hw_accel ? joinpath(RECORD_DIR, "hw") : RECORD_DIR
record_scores(hw_accel::Bool) = joinpath(RECORD_DIR, hw_accel ? "scores_hw.csv" : "scores_sw.csv")

# ── Recording ───────────────────────────────────────────────────────────────
# The test run is what the gallery reports on. It writes every render and every
# metric it just asserted against, and `run_comparison.jl` does nothing but
# present that record.
#
# A gallery that renders its own copy, with its render step and its tonemap
# step each beginning `isfile(...) && continue`, makes `recorded/`
# write-once and froze at whatever was rendered first. The committed gallery was
# months older than the renderer and no change could ever reach it. Writes here
# are unconditional for exactly that reason.

"""Start a fresh record for one backend, discarding the previous run's scores."""
function begin_record!(hw_accel::Bool)
    mkpath(record_dir(hw_accel))
    open(record_scores(hw_accel), "w") do io
        println(io, "name,samples,tile,energy,mean_diff,median_diff,max_diff")
    end
    return nothing
end

"""Record one scene's render and the metrics the suite just checked."""
function record_scene!(scene_name::AbstractString, fb, m; hw_accel::Bool, samples::Int)
    dir = record_dir(hw_accel)
    mkpath(dir)
    FileIO.save(joinpath(dir, "$(scene_name).exr"), fb)
    open(record_scores(hw_accel), "a") do io
        println(io, join((scene_name, samples, m.tile, m.energy_ratio,
                          m.mean_diff, m.median_diff, m.max_diff), ","))
    end
    return nothing
end
