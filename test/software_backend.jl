# A CPU Vulkan driver exercises the same Mantle graph on the host processor.
# KA.CPU() is POCL in this workspace, which does not implement Mantle graphs.
if !@isdefined(software_reference_backend)
    function software_reference_backend()
        infos = Mantle.devices()
        index = findfirst(info -> info.kind === :cpu, infos)
        return index === nothing ? nothing : Mantle.Device(infos[index].index)
    end
end
