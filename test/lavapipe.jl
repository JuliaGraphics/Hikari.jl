# Opt in before Hikari creates its first device. Loading Lavapipe_jll adds its driver
# to the Vulkan loader's search path; selecting it by name keeps a runner's hardware
# driver from being picked instead.
if get(ENV, "HIKARI_TEST_LAVAPIPE", "false") == "true"
    using Lavapipe_jll, Mantle
    Mantle.defaultdevice!("lavapipe")
end
