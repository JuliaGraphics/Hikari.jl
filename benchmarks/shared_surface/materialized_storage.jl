# Isolate the completed value from setup's temporary mutable storage. This is
# an invocation-local value boundary, not a second graph evaluation or buffer.
Base.@constprop :none @noinline surface_ready(lobes) = lobes
