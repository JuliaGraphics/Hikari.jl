# Run in an environment containing this Hikari checkout. The default renderer
# selection is unchanged; renderer.jl installs its override only within compare.
using Hikari
Base.include(Hikari, joinpath(@__DIR__, "storage.jl"))
Base.include(Hikari, joinpath(@__DIR__, "program.jl"))
include("cpu.jl")
include("classes.jl")
include("renderer.jl")
include("kernel.jl")
