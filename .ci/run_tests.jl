module HikariCITests
    using Pkg
    include(joinpath(@__DIR__, "..", "test", "test_files.jl"))

    function test_group(group::String; groups::Int = 4)
        group == "pbrt" && return ["pbrt"]
        index = parse(Int, group)
        1 <= index <= groups || throw(ArgumentError("test group must be between 1 and $groups, or pbrt"))
        return TEST_FILES[index:groups:end]
    end

    function run_tests(group::String)
        return Pkg.test(
            "Hikari"; allow_reresolve = false,
            test_args = test_group(group), julia_args = `--check-bounds=auto`
        )
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    HikariCITests.run_tests(only(ARGS))
end
