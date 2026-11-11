"""
What Hikari still names from Lava, and why each one is allowed to stay.

`Lava.` in Hikari's source means a renderer reaching into a Vulkan runtime, and
this file is what keeps the count at zero.

Lava is a Julia→SPIR-V compiler with no Vulkan dependency, and everything a
renderer would reach for (`LavaBackend`, `VulkanTLAS`, `vk_context`,
`@compile_workload`) is Mantle's. Hikari traces with `Raycore.closest_hit` from
ordinary kernels, which Mantle lowers to a ray query, so it needs no device-side
name from Lava either. The count is **zero names in zero files**, and any
reference at all is the split coming undone.

Parsed rather than grepped — `import Lava: a, b,` continues across lines, and a
regex either misses the continuation or matches the word in a comment.
"""

using Test

# Empty, and it stays empty. A file appearing here is a name Hikari took from
# Lava, which can only be a runtime reference: the shader vocabulary is
# `KernelInterface`'s and reaches Hikari through Mantle.
const ALLOWED_LAVA_REFS = Dict{String, Dict{String, Int}}()

"""
Every `Lava.<name>` and every name in an `import Lava: …` list, as `name =>
count`.

`Lava` alone — `import Lava`, and the qualifier of a counted `Lava.foo` — is not
a reference to anything. What matters is which of Lava's names Hikari knows.
"""
function lava_references(path::AbstractString)
    found = Dict{String, Int}()
    note!(name) = (found[name] = get(found, name, 0) + 1)

    function walk(ex)
        ex isa Expr || return
        if ex.head === :. && length(ex.args) == 2 &&
                ex.args[1] === :Lava && ex.args[2] isa QuoteNode
            note!(String(ex.args[2].value))
            return
        end
        if (ex.head === :import || ex.head === :using) && length(ex.args) == 1 &&
                ex.args[1] isa Expr && ex.args[1].head === :(:)
            spec = ex.args[1]
            if spec.args[1] isa Expr && spec.args[1].head === :. &&
                    spec.args[1].args == [:Lava]
                for name in spec.args[2:end]
                    name isa Expr && name.head === :. && note!(String(name.args[1]))
                end
                return
            end
        end
        foreach(walk, ex.args)
        return
    end

    walk(Meta.parseall(read(path, String)))
    return found
end

@testset "Hikari names only shader vocabulary from Lava" begin
    src = joinpath(pkgdir(Hikari), "src")

    actual = Dict{String, Dict{String, Int}}()
    for (root, _, files) in walkdir(src), file in files
        endswith(file, ".jl") || continue
        path = joinpath(root, file)
        refs = lava_references(path)
        isempty(refs) || (actual[relpath(path, pkgdir(Hikari))] = refs)
    end

    # ANY file naming Lava is the case this catches.
    isempty(actual) ||
        @info "Hikari names Lava" files = sort(collect(keys(actual))) refs = actual
    @test sort(collect(keys(actual))) == sort(collect(keys(ALLOWED_LAVA_REFS)))

    for (file, allowed) in ALLOWED_LAVA_REFS
        @testset "$file" begin
            refs = get(actual, file, Dict{String, Int}())
            @test isempty(setdiff(keys(refs), keys(allowed)))
            gone = setdiff(keys(allowed), keys(refs))
            isempty(gone) ||
                @info "$file no longer references $(join(sort(collect(gone)), ", ")) — delete these from ALLOWED_LAVA_REFS"
            @test isempty(gone)
            for (name, n) in allowed
                @test get(refs, name, 0) == n
            end
        end
    end

    # And the ledger itself is empty — a line added back here has to be argued
    # for in this file rather than slipped in beside thirteen others.
    @test isempty(ALLOWED_LAVA_REFS)
end
