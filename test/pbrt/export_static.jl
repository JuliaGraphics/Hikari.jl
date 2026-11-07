# ============================================================================
# Export the Hikari vs pbrt-v4 comparison app as a static HTML site
#
# Usage:
#   julia --project=/sim/Programmieren/VulkanDev export_static.jl
#
# Produces: build/ directory with index.html + assets
# Can be deployed to GitHub Pages or served by any static file server.
# ============================================================================

using Bonito, DelimitedFiles, FileIO, Statistics, Colors

const PBRT_DIR     = @__DIR__
const REFS_DIR     = joinpath(PBRT_DIR, "references")
const RECORDED_DIR = joinpath(PBRT_DIR, "recorded")
const DISPLAY_DIR  = joinpath(PBRT_DIR, "display")
const BUILD_DIR    = joinpath(PBRT_DIR, "build")
const SPP          = 256

# ============================================================================
# Reuse tile_score and tonemap from comparison_app.jl
# ============================================================================

function downsample_2x(img)
    h, w = size(img)
    h2, w2 = h ÷ 2, w ÷ 2
    out = similar(img, h2, w2)
    for i in 1:h2, j in 1:w2
        p1 = img[2i-1, 2j-1]; p2 = img[2i, 2j-1]; p3 = img[2i-1, 2j]; p4 = img[2i, 2j]
        r = (Float64(red(p1)) + Float64(red(p2)) + Float64(red(p3)) + Float64(red(p4))) / 4
        g = (Float64(green(p1)) + Float64(green(p2)) + Float64(green(p3)) + Float64(green(p4))) / 4
        b = (Float64(blue(p1)) + Float64(blue(p2)) + Float64(blue(p3)) + Float64(blue(p4))) / 4
        out[i, j] = typeof(p1)(r, g, b)
    end
    return out
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
    scores[clamp(ceil(Int, percentile * length(scores)), 1, length(scores))]
end

function tonemap_to_png(exr_path, png_path)
    isfile(png_path) && return
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
# Compute scores + ensure PNGs exist
# ============================================================================

println("Computing scores from saved EXRs...")
scene_files = sort(filter(f -> endswith(f, ".pbrt"), readdir(joinpath(PBRT_DIR, "scenes"))))
names = String[]
scores = Float64[]

for fname in scene_files
    name = replace(fname, ".pbrt" => "")
    ref_exr = joinpath(REFS_DIR, "$(name).exr")
    rec_exr = joinpath(RECORDED_DIR, "$(name).exr")
    (isfile(ref_exr) && isfile(rec_exr)) || continue
    ref_img = FileIO.load(ref_exr)
    rec_img = FileIO.load(rec_exr)
    push!(names, name)
    push!(scores, tile_score(ref_img, rec_img))
    tonemap_to_png(ref_exr, joinpath(DISPLAY_DIR, "ref_$(name).png"))
    tonemap_to_png(rec_exr, joinpath(DISPLAY_DIR, "rec_$(name).png"))
end

# Sort worst first
order = sortperm(scores, rev=true)
names = names[order]
scores = scores[order]

n_pass = count(s -> s < 0.07, scores)
println("$(n_pass)/$(length(scores)) pass at $(SPP) spp")

# ============================================================================
# Build Bonito app (same as comparison_app.jl)
# ============================================================================

import Bonito: DOM

app = App() do session
    cards_vec = Any[]
    for (name, score) in zip(names, scores)
        short = replace(name, "mat_" => "", "_light_" => " + ")
        rec_path = Bonito.Asset(joinpath(DISPLAY_DIR, "rec_$(name).png"))
        ref_path = Bonito.Asset(joinpath(DISPLAY_DIR, "ref_$(name).png"))
        score_color = score > 0.10 ? "#E74C3C" : score > 0.07 ? "#F39C12" : score > 0.03 ? "#FDD835" : "#4CAF50"
        border_color = score > 0.07 ? score_color : "#333"
        card_id = "c$(hash(name) & 0xFFFF)"
        card = DOM.div(
            DOM.h3(short, style="margin:4px 0; font-size:14px; color:white;"),
            DOM.div("Score: $(round(score, digits=4))", style="color:$score_color; font-weight:bold; margin-bottom:4px;"),
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
    n_pass = count(s -> s < 0.07, scores)
    n_fail = count(s -> s >= 0.07, scores)
    DOM.div(
        DOM.h1("Hikari vs pbrt-v4 ($(SPP) spp)", style="color:white; text-align:center; margin:8px;"),
        DOM.div("$(length(scores)) tests | $(n_pass) pass | $(n_fail) fail — click to toggle",
            style="color:#aaa; text-align:center; margin-bottom:16px;"),
        DOM.div(cards_vec..., style="display:flex; flex-wrap:wrap; justify-content:center;"),
        style="background:#111; padding:16px; font-family:monospace;"
    )
end

# ============================================================================
# Export static site
# ============================================================================

mkpath(BUILD_DIR)
routes = Bonito.Routes("/" => app)
println("Exporting static site to $BUILD_DIR...")
Bonito.export_static(BUILD_DIR, routes)
println("Done! Open $(joinpath(BUILD_DIR, "index.html")) in a browser.")
