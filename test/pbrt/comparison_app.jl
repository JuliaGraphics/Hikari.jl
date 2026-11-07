# ============================================================================
# Bonito comparison app — Hikari vs pbrt-v4 visual reference tests
#
# Design: all state lives on disk, no re-rendering needed after restart.
#   references/<name>.exr  — pbrt-v4 output (generated if missing)
#   recorded/<name>.exr    — Hikari output (generated if missing)
#   display/ref_<name>.png — tonemapped pbrt for web display
#   display/rec_<name>.png — tonemapped Hikari for web display
#
# Usage:
#   include("comparison_app.jl")
#   # Renders only missing scenes, then opens http://localhost:9384
#   # Click any card to toggle between Hikari and pbrt-v4
# ============================================================================

using Hikari, Lava, GeometryBasics, LinearAlgebra
using FileIO, Statistics, Colors, DelimitedFiles, Bonito, ImageFiltering
import KernelAbstractions as KA
import Bonito: DOM

const PBRT_DIR     = @__DIR__
const SCENES_DIR   = joinpath(PBRT_DIR, "scenes")
const REFS_DIR     = joinpath(PBRT_DIR, "references")
const RECORDED_DIR = joinpath(PBRT_DIR, "recorded")
const DISPLAY_DIR  = joinpath(PBRT_DIR, "display")
const PBRT_BIN     = "/sim/Programmieren/VulkanDev/pbrt-v4/build/pbrt"
const SPP          = 256

mkpath(REFS_DIR)
mkpath(RECORDED_DIR)
mkpath(DISPLAY_DIR)

# ============================================================================
# Tonemap HDR EXR → sRGB PNG for web display
# ============================================================================

function tonemap_to_png(exr_path, png_path)
    isfile(png_path) && return  # skip if already exists
    img = FileIO.load(exr_path)
    h, w = size(img)
    out = Array{RGB{Float64}}(undef, h, w)
    for i in eachindex(img)
        p = img[i]
        r = Float64(red(p)); g = Float64(green(p)); b = Float64(blue(p))
        r = r / (1.0 + r); g = g / (1.0 + g); b = b / (1.0 + b)
        γ(x) = x <= 0.0031308 ? 12.92x : 1.055 * x^(1/2.4) - 0.055
        out[i] = RGB(γ(clamp(r,0,1)), γ(clamp(g,0,1)), γ(clamp(b,0,1)))
    end
    FileIO.save(png_path, out)
end

# ============================================================================
# Log-space tile p95 score (same as test_pbrt_all_materials.jl)
# ============================================================================

function downsample_2x(img)
    blurred = imfilter(img, Kernel.gaussian((0.75, 0.75)))
    return blurred[1:2:end, 1:2:end]
end

function tile_score(a::AbstractMatrix, b::AbstractMatrix; tile_size=16, percentile=0.95)
    size(a) != size(b) && return Inf
    a = downsample_2x(a); b = downsample_2x(b)
    h, w = size(a)
    range_h = round.(Int, range(0, h, length=max(2, ceil(Int, h / tile_size))))
    range_w = round.(Int, range(0, w, length=max(2, ceil(Int, w / tile_size))))
    boundary_iter(boundaries) = zip(boundaries[1:end-1] .+ 1, boundaries[2:end])
    function log_color_dist(p1, p2)
        r1 = log1p(max(0.0, Float64(red(p1)))); g1 = log1p(max(0.0, Float64(green(p1)))); b1 = log1p(max(0.0, Float64(blue(p1))))
        r2 = log1p(max(0.0, Float64(red(p2)))); g2 = log1p(max(0.0, Float64(green(p2)))); b2 = log1p(max(0.0, Float64(blue(p2))))
        return sqrt((r1-r2)^2 + (g1-g2)^2 + (b1-b2)^2)
    end
    scores = [mean(log_color_dist.(@view(a[r1:r2, c1:c2]), @view(b[r1:r2, c1:c2])))
              for (r1, r2) in boundary_iter(range_h) for (c1, c2) in boundary_iter(range_w)]
    sort!(scores)
    idx = clamp(ceil(Int, percentile * length(scores)), 1, length(scores))
    return scores[idx]
end

# ============================================================================
# Step 1: Ensure all pbrt references exist
# ============================================================================

function ensure_pbrt_references(; spp=SPP)
    scene_files = sort(filter(f -> endswith(f, ".pbrt"), readdir(SCENES_DIR)))
    n_generated = 0
    for fname in scene_files
        name = replace(fname, ".pbrt" => "")
        ref_exr = joinpath(REFS_DIR, "$(name).exr")
        isfile(ref_exr) && continue
        scene_file = joinpath(SCENES_DIR, fname)
        println("  Generating pbrt reference: $name ($spp spp)")
        run(`$PBRT_BIN $scene_file --outfile $ref_exr --spp $spp`)
        n_generated += 1
    end
    n_generated > 0 && println("  Generated $n_generated new pbrt references")
end

# ============================================================================
# Step 2: Render missing Hikari scenes
# ============================================================================

function render_missing_hikari(; spp=SPP)
    backend = KA.CPU()

    scene_files = sort(filter(f -> endswith(f, ".pbrt"), readdir(SCENES_DIR)))
    n_rendered = 0
    for (i, fname) in enumerate(scene_files)
        name = replace(fname, ".pbrt" => "")
        rec_exr = joinpath(RECORDED_DIR, "$(name).exr")
        isfile(rec_exr) && continue

        scene_file = joinpath(SCENES_DIR, fname)
        ref_exr = joinpath(REFS_DIR, "$(name).exr")
        isfile(ref_exr) || continue

        try
            fb = Array(Hikari.render_pbrt(scene_file; backend=backend, samples=spp))
            FileIO.save(rec_exr, fb)
            n_rendered += 1
            println("  Rendered $i/$(length(scene_files)) $name")
        catch e
            println("  SKIP $i/$(length(scene_files)) $name: $(sprint(showerror, e; context=:limit=>200))")
        end
    end
    n_rendered > 0 && println("  Rendered $n_rendered new Hikari scenes")
end

# ============================================================================
# Step 3: Compute scores from saved EXRs + generate display PNGs
# ============================================================================

function compute_energy_ratio(ref_img, rec_img)
    total_ref = 0.0
    total_hik = 0.0
    for i in eachindex(ref_img)
        rp = ref_img[i]; hp = rec_img[i]
        total_ref += Float64(red(rp)) + Float64(green(rp)) + Float64(blue(rp))
        total_hik += Float64(red(hp)) + Float64(green(hp)) + Float64(blue(hp))
    end
    return total_hik / max(total_ref, 1e-10)
end

function compute_scores_and_pngs()
    scene_files = sort(filter(f -> endswith(f, ".pbrt"), readdir(SCENES_DIR)))
    scores = Float64[]
    energies = Float64[]
    names = String[]

    for fname in scene_files
        name = replace(fname, ".pbrt" => "")
        ref_exr = joinpath(REFS_DIR, "$(name).exr")
        rec_exr = joinpath(RECORDED_DIR, "$(name).exr")
        (isfile(ref_exr) && isfile(rec_exr)) || continue

        ref_img = FileIO.load(ref_exr)
        rec_img = FileIO.load(rec_exr)
        sc = tile_score(ref_img, rec_img)
        en = compute_energy_ratio(ref_img, rec_img)
        push!(scores, sc)
        push!(energies, en)
        push!(names, name)

        # Generate display PNGs (skips if already exist)
        tonemap_to_png(ref_exr, joinpath(DISPLAY_DIR, "ref_$(name).png"))
        tonemap_to_png(rec_exr, joinpath(DISPLAY_DIR, "rec_$(name).png"))
    end
    return names, scores, energies
end

# ============================================================================
# Bonito app — click cards to toggle Hikari/pbrt-v4
# ============================================================================

function make_comparison_app(names, scores, energies; spp=SPP)
    App() do session
        cards_vec = Any[]
        for (name, score, energy) in zip(names, scores, energies)
            short = replace(name, "mat_" => "", "_light_" => " + ")
            rec_path = Bonito.Asset(joinpath(DISPLAY_DIR, "rec_$(name).png"))
            ref_path = Bonito.Asset(joinpath(DISPLAY_DIR, "ref_$(name).png"))
            energy_ok = 0.95 < energy < 1.05
            score_color = !energy_ok ? "#E74C3C" : score > 0.10 ? "#F39C12" : score > 0.03 ? "#FDD835" : "#4CAF50"
            border_color = (!energy_ok || score > 0.10) ? score_color : "#333"
            energy_color = energy_ok ? "#aaa" : "#E74C3C"
            card_id = "c$(hash(name) & 0xFFFF)"
            card = DOM.div(
                DOM.h3(short, style="margin:4px 0; font-size:14px; color:white;"),
                DOM.div(
                    DOM.span("tile: $(round(score, digits=3))", style="color:$score_color;"),
                    DOM.span(" | ", style="color:#555;"),
                    DOM.span("energy: $(round(energy, digits=3))", style="color:$energy_color;"),
                    style="font-weight:bold; margin-bottom:4px; font-size:12px;"
                ),
                DOM.div(
                    DOM.img(src=rec_path, id="$(card_id)_rec", style="width:100%; image-rendering:pixelated; position:absolute; top:0; left:0; z-index:2; cursor:pointer;"),
                    DOM.img(src=ref_path, style="width:100%; image-rendering:pixelated; position:absolute; top:0; left:0; z-index:1;"),
                    style="position:relative; width:100%; padding-bottom:100%; overflow:hidden;"
                ),
                DOM.div("Hikari", id="$(card_id)_label", style="text-align:center; font-size:12px; color:#aaa; margin-top:4px;"),
                DOM.script("""(function(){var r=document.getElementById('$(card_id)_rec'),l=document.getElementById('$(card_id)_label'),s='hikari';r.parentElement.addEventListener('click',function(){if(s==='hikari'){r.style.opacity='0';l.textContent='pbrt-v4';s='pbrt';}else{r.style.opacity='1';l.textContent='Hikari';s='hikari';}});})();"""),
                style="border:2px solid $border_color; border-radius:8px; padding:8px; margin:4px; background:#1a1a1a; width:250px;"
            )
            push!(cards_vec, card)
        end
        n_energy_ok = Base.count(e -> 0.95 < e < 1.05, energies)
        n_tile_ok = Base.count(s -> s < 0.07, scores)
        DOM.div(
            DOM.h1("Hikari vs pbrt-v4 ($(spp) spp)", style="color:white; text-align:center; margin:8px;"),
            DOM.div("$(length(scores)) scenes | energy within 5%: $(n_energy_ok)/$(length(scores)) | tile<0.07: $(n_tile_ok)/$(length(scores)) -- click to toggle",
                style="color:#aaa; text-align:center; margin-bottom:16px;"),
            DOM.div(cards_vec..., style="display:flex; flex-wrap:wrap; justify-content:center;"),
            style="background:#111; padding:16px; font-family:monospace;"
        )
    end
end

# ============================================================================
# Main
# ============================================================================

println("Step 1: Checking pbrt references...")
ensure_pbrt_references(; spp=SPP)

println("Step 2: Rendering missing Hikari scenes...")
render_missing_hikari(; spp=SPP)

println("Step 3: Computing scores from saved EXRs...")
all_names, all_scores, all_energies = compute_scores_and_pngs()

# Sort by worst energy deviation first, then by tile score
order = sortperm(1:length(all_names); by=i -> (-abs(all_energies[i] - 1.0), -all_scores[i]))
all_names = all_names[order]
all_scores = all_scores[order]
all_energies = all_energies[order]

n_energy_ok = Base.count(e -> 0.95 < e < 1.05, all_energies)
println("Step 4: Launching Bonito app...")
server = Bonito.Server(make_comparison_app(all_names, all_scores, all_energies; spp=SPP), "0.0.0.0", 9384)
println("http://localhost:9384 -- energy within 5%: $(n_energy_ok)/$(length(all_scores)) at $(SPP) spp")
