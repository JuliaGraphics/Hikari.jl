# Large files live outside the repository: as artifacts (tarballs attached to the `assets`
# release, bound by content hash in Artifacts.toml) or, for the README's images, as plain
# files of that release. Publishing needs the `gh` CLI with write access to the repository.
#
#     include("assets/publish.jl")
#     publish_artifact("pbrt_references", "test/pbrt/references"; lazy = true)
#
# then commit the changed Artifacts.toml.

using Pkg.Artifacts

const HIKARI_DIR = dirname(@__DIR__)

"""
    release_url(file; repo = "JuliaGraphics/Hikari.jl", release = "assets")

Download URL of `file` in the `release` of `repo`.
"""
release_url(file; repo = "JuliaGraphics/Hikari.jl", release = "assets") =
    "https://github.com/$repo/releases/download/$release/$file"

"""
    upload_release_file(path; repo, release)

Attach `path` to `release`, replacing a file of the same name.
"""
function upload_release_file(path; repo = "JuliaGraphics/Hikari.jl", release = "assets")
    run(`gh release upload $release $path --repo $repo --clobber`)
    return release_url(basename(path); repo, release)
end

"""
    publish_artifact(name, dir; lazy, toml = Artifacts.toml at the package root, repo, release)

Pack the contents of `dir` into the artifact `name`, upload its tarball (named by its content
hash, so older commits keep finding theirs) and bind it in `toml`. Returns the tree hash.
"""
function publish_artifact(name, dir; lazy::Bool, toml = joinpath(HIKARI_DIR, "Artifacts.toml"),
                          repo = "JuliaGraphics/Hikari.jl", release = "assets")
    tree = create_artifact() do target
        for entry in readdir(dir)
            cp(joinpath(dir, entry), joinpath(target, entry))
        end
    end
    tarball = joinpath(mktempdir(), "$name-$(bytes2hex(tree.bytes)).tar.gz")
    sha256 = archive_artifact(tree, tarball)
    url = upload_release_file(tarball; repo, release)
    bind_artifact!(toml, name, tree; download_info = Tuple[(url, sha256)], lazy, force = true)
    return tree
end
