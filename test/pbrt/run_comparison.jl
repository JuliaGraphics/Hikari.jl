#!/usr/bin/env julia
#
# Build the pbrt comparison gallery from a recorded test run.
#
# This script renders nothing and decides nothing. The test suite
# (`test_pbrt_all_materials.jl`, driven by `test/runtests.jl`) is the single
# thing that renders scenes, compares them against the pbrt-v4 references and
# decides pass/fail; as it goes it records each render and each metric under
# `recorded/`. This reads that record and presents it.
#
# A gallery that renders, thresholds and decides pass/fail itself comes apart
# from the suite in three ways:
#
#   * Both its render step and its tonemap step opened with
#     `isfile(...) && continue`, so `recorded/` and `display/` were write-once.
#     The committed gallery was months older than the renderer and no amount of
#     rendering work could ever reach it.
#   * It applied a looser band below 128 spp (tile 0.10, energy ±10 %) and
#     defaulted to 32 spp, so it ran the relaxed band essentially always.
#   * It carried a SCENE_OVERRIDES table reaching tile=0.55 for scenes the suite
#     checks at 0.07, kept alive by a comment noting the underlying bugs had
#     since been fixed.
#
# Inputs (written by the suite):
#   recorded/<name>.exr           - SW renders
#   recorded/hw/<name>.exr        - HW RT renders
#   recorded/scores_{sw,hw}.csv   - the metrics the suite asserted on
#
# Outputs (all under test/pbrt/display/):
#   ref_<name>.png                - tonemapped pbrt-v4 reference
#   rec_<name>.png / rec_hw_...   - tonemapped Hikari output
#   gallery.html / gallery_hw.html
#   scores_{sw,hw}.csv            - copied alongside, so display/ is self-contained
#
# Usage:  julia --project=test/pbrt test/pbrt/run_comparison.jl
# Exit code is 0 whenever a record could be presented; the suite reports failures.

using FileIO, Colors

include(joinpath(@__DIR__, "suite.jl"))

const REFS_DIR    = joinpath(@__DIR__, "references")
const DISPLAY_DIR = joinpath(@__DIR__, "display")

struct SceneRow
    name::String
    samples::Int
    tile::Float64
    energy::Float64
end

# ── Reading the record ──────────────────────────────────────────────────────

function read_record(hw_accel::Bool)
    path = record_scores(hw_accel)
    isfile(path) || return SceneRow[]
    rows = SceneRow[]
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue                      # header
        isempty(strip(line)) && continue
        f = split(line, ',')
        length(f) >= 4 || error("$(path):$(i): expected at least 4 fields, got $(length(f))")
        push!(rows, SceneRow(f[1], parse(Int, f[2]), parse(Float64, f[3]), parse(Float64, f[4])))
    end
    return rows
end

# ── Tonemapping ─────────────────────────────────────────────────────────────

"""sRGB-encode an EXR and write it as a PNG. Always rewrites: a stale PNG next
to a fresh render is the exact failure this script was rebuilt to remove."""
function tonemap_to_png(exr_path, png_path)
    img = FileIO.load(exr_path)
    out = similar(img, RGB{Float64})
    gamma(x) = x <= 0.0031308 ? 12.92x : 1.055 * x^(1 / 2.4) - 0.055
    for i in eachindex(img)
        p = img[i]
        out[i] = RGB{Float64}(gamma(clamp(Float64(red(p)),   0, 1)),
                              gamma(clamp(Float64(green(p)), 0, 1)),
                              gamma(clamp(Float64(blue(p)),  0, 1)))
    end
    FileIO.save(png_path, out)
    return nothing
end

# ── Gallery ─────────────────────────────────────────────────────────────────

function build_gallery(rows::Vector{SceneRow}, hw_accel::Bool)
    label     = hw_accel ? "HW RT" : "SW BVH"
    rec_pfx   = hw_accel ? "rec_hw_" : "rec_"
    html_name = hw_accel ? "gallery_hw.html" : "gallery.html"
    dir       = record_dir(hw_accel)

    # Worst first: the scenes worth looking at are the ones nearest the limits.
    sorted = sort(rows; by = s -> (-abs(s.energy - 1.0), -s.tile))

    entries = String[]
    for s in sorted
        ref_exr = joinpath(REFS_DIR, "$(s.name).exr")
        rec_exr = joinpath(dir, "$(s.name).exr")
        if !(isfile(ref_exr) && isfile(rec_exr))
            println("  skipping $(s.name): missing $(isfile(ref_exr) ? "render" : "reference")")
            continue
        end
        tonemap_to_png(ref_exr, joinpath(DISPLAY_DIR, "ref_$(s.name).png"))
        tonemap_to_png(rec_exr, joinpath(DISPLAY_DIR, "$(rec_pfx)$(s.name).png"))
        push!(entries, """{"name":"$(s.name)","tile":$(round(s.tile, digits=4)),"energy":$(round(s.energy, digits=4)),"ref":"ref_$(s.name).png","rec":"$(rec_pfx)$(s.name).png"}""")
    end
    isempty(entries) && return 0

    samples = isempty(sorted) ? 0 : first(sorted).samples
    n_pass  = count(s -> within_tolerance(s.tile, s.energy), sorted)

    # Thresholds are interpolated from `suite.jl` rather than written out again —
    # the JS reading different numbers than the suite is how the two drift.
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Hikari vs pbrt-v4 — $(label)</title>
<style>
  body { margin:0; background:#111; font-family:monospace; }
  #header { color:#aaa; text-align:center; padding:12px 0 8px; font-size:13px; }
  #grid { display:flex; flex-wrap:wrap; justify-content:center; padding:0 8px 16px; }
  .card { border-radius:8px; padding:8px; margin:4px; background:#1a1a1a; width:220px; }
  .card-title { margin:4px 0; font-size:13px; color:white; font-weight:bold; word-break:break-all; }
  .card-scores { font-size:11px; margin-bottom:4px; }
  .card-img { position:relative; width:100%; padding-bottom:100%; overflow:hidden; cursor:pointer; }
  .card-img img { width:100%; position:absolute; top:0; left:0; image-rendering:pixelated; }
  .card-label { text-align:center; font-size:11px; color:#aaa; margin-top:4px; }
</style>
</head>
<body>
<div id="header">Loading...</div>
<div id="grid"></div>
<script>
var TILE_MAX = $(TILE_THRESHOLD), E_LOW = $(ENERGY_LOW), E_HIGH = $(ENERGY_HIGH);
var scenes = [
$(join(entries, ",\n"))
];
document.getElementById('header').textContent =
  '$(label), $(samples) spp | ' + scenes.length + ' scenes | within tolerance (tile<' +
  TILE_MAX + ', energy ' + E_LOW + '-' + E_HIGH + '): $(n_pass)/' + scenes.length +
  ' | click an image to toggle Hikari/pbrt-v4';

var grid = document.getElementById('grid');
for (var i = 0; i < scenes.length; i++) {
  (function(s) {
    var eok = s.energy > E_LOW && s.energy < E_HIGH;
    var sc = !eok ? '#E74C3C' : s.tile > TILE_MAX ? '#F39C12' : s.tile > TILE_MAX / 2 ? '#FDD835' : '#4CAF50';
    var bc = (!eok || s.tile > TILE_MAX) ? sc : '#333';
    var ec = eok ? '#aaa' : '#E74C3C';
    var short = s.name.replace('mat_', '').replace(/_light_/g, ' + ');

    var card = document.createElement('div');
    card.className = 'card';
    card.style.border = '2px solid ' + bc;

    var title = document.createElement('div');
    title.className = 'card-title';
    title.textContent = short;

    var scores = document.createElement('div');
    scores.className = 'card-scores';
    var tileSpan = document.createElement('span');
    tileSpan.style.color = sc;
    tileSpan.textContent = 'tile: ' + s.tile.toFixed(3);
    var sep = document.createElement('span');
    sep.style.color = '#555';
    sep.textContent = ' | ';
    var energySpan = document.createElement('span');
    energySpan.style.color = ec;
    energySpan.textContent = 'energy: ' + s.energy.toFixed(3);
    scores.appendChild(tileSpan);
    scores.appendChild(sep);
    scores.appendChild(energySpan);

    var wrap = document.createElement('div');
    wrap.className = 'card-img';
    var imgRec = document.createElement('img');
    imgRec.src = s.rec;
    imgRec.style.zIndex = '2';
    var imgRef = document.createElement('img');
    imgRef.src = s.ref;
    imgRef.style.zIndex = '1';
    wrap.appendChild(imgRec);
    wrap.appendChild(imgRef);

    var lbl = document.createElement('div');
    lbl.className = 'card-label';
    lbl.textContent = 'Hikari';

    wrap.addEventListener('click', function(rec, l) {
      return function() {
        if (l.textContent === 'Hikari') { rec.style.opacity = '0'; l.textContent = 'pbrt-v4'; }
        else { rec.style.opacity = '1'; l.textContent = 'Hikari'; }
      };
    }(imgRec, lbl));

    card.appendChild(title);
    card.appendChild(scores);
    card.appendChild(wrap);
    card.appendChild(lbl);
    grid.appendChild(card);
  })(scenes[i]);
}
</script>
</body>
</html>
"""
    write(joinpath(DISPLAY_DIR, html_name), html)
    cp(record_scores(hw_accel),
       joinpath(DISPLAY_DIR, basename(record_scores(hw_accel))); force = true)
    println("  $(label): $(length(entries)) scenes, $(n_pass) within tolerance -> $(html_name)")
    return length(entries)
end

# ── Main ────────────────────────────────────────────────────────────────────

mkpath(DISPLAY_DIR)

# Clear the previous page before building this one. Anything left behind is a
# scene that is not in the record, and a directory holding some images from this
# run and some from an earlier one is the failure mode this
# exists to remove — `display/` is an output, not an accumulator.
for f in readdir(DISPLAY_DIR)
    if startswith(f, "ref_") || startswith(f, "rec_") ||
       startswith(f, "gallery") || startswith(f, "scores")
        rm(joinpath(DISPLAY_DIR, f))
    end
end

total = 0
for hw_accel in (false, true)
    rows = read_record(hw_accel)
    if isempty(rows)
        println("No $(hw_accel ? "HW RT" : "SW BVH") record at $(record_scores(hw_accel)) — skipping.")
        continue
    end
    global total += build_gallery(rows, hw_accel)
end

if total == 0
    error("""
          No recorded run to present.

          The gallery is built from what the test suite recorded, so run the
          suite first:

              julia --project=. dev/Hikari/test/runtests.jl

          That writes recorded/<name>.exr and recorded/scores_sw.csv (plus
          recorded/hw/ when HW RT runs), which this script turns into
          display/gallery.html.
          """)
end
println("Gallery written to $(DISPLAY_DIR)")
