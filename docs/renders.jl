# The docs' images are rendered once, at full quality, and published as Hikari's
# `docs_renders` artifact. A normal docs build runs no example: it shows each one's code and
# the published image (`PublishedExample`), and takes a minute where running them on
# Lavapipe took CI 34. `HIKARI_DOCS_EXAMPLES=true` runs them, at one sample per pixel to prove
# the code works, still showing the published images; CI's Docs Examples job does that.
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

using Downloads, FileIO, SHA, TOML
using Base64: base64encode
import Documenter
using Pkg.Artifacts: ensure_artifact_installed

render_mode() = get(ENV, "HIKARI_DOCS_RENDER", "false") == "true"

"""The Makie logo ball of the material renders, made by `docs/utils/logo_ball`."""
logo_ball_path() = joinpath(ensure_artifact_installed("logo_ball", joinpath(@__DIR__, "..", "Artifacts.toml")), "logo_ball.obj")

"""Whether this build runs the examples: to render the images, or to check that they run."""
run_examples() = render_mode() || get(ENV, "HIKARI_DOCS_EXAMPLES", "false") == "true"

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

# A build that runs no example shows each `@example` block as its code and the published image
# its `docs_image` names, and a `@setup` block as nothing. Every example block ends in a
# `docs_image` or in `nothing # hide`, so that is everything a block would show. Runs ahead of
# Documenter's own `ExampleBlocks` (8.0) and `SetupBlocks` (10.0), which then find no block left.
abstract type PublishedExample <: Documenter.Expanders.ExpanderPipeline end
Documenter.Selectors.order(::Type{PublishedExample}) = 7.5
Documenter.Selectors.disable(::Type{PublishedExample}) = run_examples()
Documenter.Selectors.matcher(::Type{PublishedExample}, node, page, doc) =
    Documenter.iscodelang(node, "@example") || Documenter.iscodelang(node, "@setup")
function Documenter.Selectors.runner(::Type{PublishedExample}, node, page, doc)
    block = node.element
    if Documenter.iscodelang(node, "@setup")
        node.element = Documenter.SetupNode(block.info, block.code)
        return
    end
    shown = [Documenter.MarkdownAST.Node(Documenter.MarkdownAST.CodeBlock("julia", Documenter.droplines(block.code)))]
    for m in eachmatch(r"docs_image\(\"([^\"]+)\"", block.code)
        png = base64encode(read(joinpath(renders_dir(), "$(m.captures[1]).png")))
        push!(shown, Documenter.MarkdownAST.Node(Documenter.MultiOutputElement(Dict{MIME, Any}(MIME"image/png"() => png))))
    end
    node.element = Documenter.MultiOutput(block)
    append!(node.children, shown)
    return
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

"""
    gallery_thumbnails(src)

The pictures of the start page's gallery. Every `/gallery/<name>-<width>.jpg` that
`index.md` shows is the render `<name>.png` of Hikari's `assets` release, shrunk to about
`width` pixels, so the page loads a few hundred kilobytes instead of the full renders it
links to. Made here, into the untracked `public/gallery`, so neither the repository nor the
release keeps a second copy; a picture made by an earlier build is kept.
"""
function gallery_thumbnails(src)
    out = mkpath(joinpath(src, "public", "gallery"))
    for m in eachmatch(r"/gallery/([A-Za-z0-9_]+)-(\d+)\.jpg", read(joinpath(src, "index.md"), String))
        name, width = m.captures[1], parse(Int, m.captures[2])
        file = joinpath(out, "$name-$width.jpg")
        isfile(file) && continue
        png = Downloads.download("https://github.com/JuliaGraphics/Hikari.jl/releases/download/assets/$name.png")
        FileIO.save(file, shrink(FileIO.load(png), width))
    end
end

# sRGB's transfer function and its inverse: `shrink` averages light, not encoded values.
linear_srgb(c::Float32) = c <= 0.04045f0 ? c / 12.92f0 : ((c + 0.055f0) / 1.055f0)^2.4f0
encode_srgb(c::Float32) = c <= 0.0031308f0 ? 12.92f0 * c : 1.055f0 * c^(1 / 2.4f0) - 0.055f0

"""
`img`, an 8-bit RGB render, made smaller by the largest whole factor that leaves it at least
`width` pixels wide: each pixel is the mean of the block of pixels it replaces.
"""
function shrink(img::AbstractMatrix, width::Int)
    f = max(1, size(img, 2) ÷ width)
    # `min`: white encodes to 1 + eps in Float32, which an 8-bit channel refuses.
    channel(block, k) = min(1f0, encode_srgb(sum(c -> linear_srgb(Float32(getfield(c, k))), block) / length(block)))
    return [begin
                block = view(img, (i - 1) * f + 1:i * f, (j - 1) * f + 1:j * f)
                eltype(img)(channel(block, :r), channel(block, :g), channel(block, :b))
            end for i in 1:size(img, 1) ÷ f, j in 1:size(img, 2) ÷ f]
end
