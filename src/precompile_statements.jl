# A no-op, so the workload above still has something to call.
#
# Generated precompile statements are worth ~22 s of inference per session in
# ten signatures, and they cannot be checked in: generated against one backend
# they name that backend's types (`LavaArray`, `LavaBackend`, `LavaDevice`,
# `MVE`, `VulkanInstanceRecord`) in a package that must name none. Regenerating
# them against whichever backend is present would be device-free by
# construction, since `precompile` takes a signature.
_precompile_statements() = nothing
