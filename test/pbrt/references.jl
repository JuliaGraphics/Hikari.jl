# Where the pbrt-v4 reference images come from, for the suite and the comparison tools.
#
# The published references are Hikari's `pbrt_references` artifact. `generate_reference`
# renders a new one into `references/` (gitignored), which wins until it is published with
#
#     include("assets/publish.jl")
#     publish_artifact("pbrt_references", merged_references(); lazy = true)
#
# and the changed Artifacts.toml is committed.

using Pkg.Artifacts: ensure_artifact_installed

const SCENES_DIR = joinpath(@__DIR__, "scenes")
const STAGED_REFS_DIR = joinpath(@__DIR__, "references")
# Beside the `dev/` checkouts by default: `VulkanDev/pbrt-v4/build/pbrt`.
const PBRT_BIN = get(ENV, "PBRT_BIN",
                     normpath(joinpath(@__DIR__, "..", "..", "..", "..", "pbrt-v4", "build", "pbrt")))

# Every reference is rendered at this sample count, and the suite verifies it
# rather than trusting it.
#
# It is deliberately a constant and not the caller's `samples`, because the
# references drift apart otherwise: of 167, eight sat at 64 spp, two at 512 and
# one at 1024 while Hikari renders all of them at 256. For the 64-spp eight that
# meant pbrt carried 4x fewer samples, so the reference was roughly 2x noisier
# than the image under test — and since the tile score is a p95 of per-pixel
# difference, that noise was scored as Hikari error. Those eight averaged tile
# 0.0169 against 0.0097 across the whole set.
#
# Keying it to `samples` instead would be worse than the bug: a CI run at 32 spp
# would quietly render every new reference at 32.
const REFERENCE_SPP = 256

# The artifact search stops at test/pbrt/Project.toml, so name Hikari's file.
published_references() = ensure_artifact_installed("pbrt_references", joinpath(@__DIR__, "..", "..", "Artifacts.toml"))

"""Path of `scene_name`'s pbrt-v4 reference: the staged render if there is one, else the published one."""
function reference_file(scene_name::AbstractString)
    staged = joinpath(STAGED_REFS_DIR, "$(scene_name).exr")
    return isfile(staged) ? staged : joinpath(published_references(), "$(scene_name).exr")
end

"""Render `scene_name`'s pbrt-v4 reference at `REFERENCE_SPP` into the staging directory."""
function generate_reference(scene_name::AbstractString)
    isfile(PBRT_BIN) || error("pbrt binary not found at $PBRT_BIN (set PBRT_BIN)")
    scene_file = joinpath(SCENES_DIR, "$(scene_name).pbrt")
    ref_file   = joinpath(mkpath(STAGED_REFS_DIR), "$(scene_name).exr")
    @info "Generating reference: $scene_name at $(REFERENCE_SPP) spp"
    # `--spp` overrides whatever `Sampler` the scene asks for, which matters:
    # six scenes pin `pixelsamples 64` in their own .pbrt.
    run(`$PBRT_BIN $scene_file --outfile $ref_file --spp $REFERENCE_SPP`)
    return ref_file
end

"""The published references with the staged ones over them, in `dir`: what to publish next."""
function merged_references(dir = mktempdir())
    for source in filter(isdir, [published_references(), STAGED_REFS_DIR]), file in readdir(source)
        cp(joinpath(source, file), joinpath(dir, file); force = true)
    end
    return dir
end
