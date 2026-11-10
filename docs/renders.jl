# The docs' images are rendered once, at full quality, and published as Hikari's
# `docs_renders` artifact. A normal docs build still runs every example, at one sample per
# pixel to prove the code works, and shows the published image instead of its own.
#
# A block shows a published image by wrapping its last expression in hidden lines:
#
#     docs_image("principled", # hide
#     showcase([...])
#     ) # hide
#
# and rendering helpers take their sample count through `docs_samples`. `make.jl` hashes the
# page code up to each `docs_image` and refuses to build when an image is older than its code.
# To render and publish the images again, on a GPU:
#
#     HIKARI_DOCS_RENDER=true julia --project=docs docs/make.jl
#     include("assets/publish.jl"); publish_artifact("docs_renders", "docs/renders"; lazy = true)
#
# then commit the changed Artifacts.toml.

using FileIO, SHA, TOML
using Pkg.Artifacts: ensure_artifact_installed

render_mode() = get(ENV, "HIKARI_DOCS_RENDER", "false") == "true"

# The artifact search stops at docs/Project.toml, so name Hikari's file.
renders_dir() = render_mode() ? mkpath(joinpath(@__DIR__, "renders")) :
                ensure_artifact_installed("docs_renders", joinpath(@__DIR__, "..", "Artifacts.toml"))

"""`samples` when rendering the published images, one otherwise."""
docs_samples(samples) = render_mode() ? samples : 1

"""The image published as `name`; when rendering them, `image`, saved under that name."""
function docs_image(name::AbstractString, image)
    file = joinpath(renders_dir(), "$name.png")
    render_mode() || return FileIO.load(file)
    FileIO.save(file, image)
    return image
end

"""The code of every `@setup` and `@example` block of the page at `path`, in order."""
docblocks(path) = [m.captures[1] for m in eachmatch(r"^```@(?:setup|example)[^\n]*\n(.*?)^```"ms, read(path, String))]

"""
    image_hashes(src) -> Dict(name => hash)

For each `docs_image` in the pages under `src`: the hash of its page's code up to and
including the block that shows it, which covers the materials and scenes defined before.
"""
function image_hashes(src)
    hashes = Dict{String, String}()
    for page in sort(filter(endswith(".md"), readdir(src)))
        code = ""
        for block in docblocks(joinpath(src, page))
            code *= block
            for m in eachmatch(r"docs_image\(\"([^\"]+)\"", block)
                name = m.captures[1]
                haskey(hashes, name) && error("docs image name \"$name\" is used twice")
                hashes[name] = bytes2hex(sha256(code))
            end
        end
    end
    return hashes
end

"""
    check_images(src)

When rendering, record the code behind each image next to it. Otherwise throw unless every
published image was rendered from the code the pages hold now.
"""
function check_images(src)
    hashes = image_hashes(src)
    manifest = joinpath(renders_dir(), "manifest.toml")
    if render_mode()
        open(io -> TOML.print(io, hashes), manifest, "w")
        return hashes
    end
    published = TOML.parsefile(manifest)
    stale = sort([name for (name, hash) in hashes if get(published, name, nothing) != hash])
    isempty(stale) || error("""
        These docs images are older than the code that shows them: $(join(stale, ", ")).
        Render and publish them again, as described in docs/renders.jl.""")
    return hashes
end
