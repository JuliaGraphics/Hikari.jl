# Opt in before Hikari/Mantle creates the first Vulkan instance. Merely adding
# this driver to the search path can still pick a runner's hardware driver.
if get(ENV, "HIKARI_TEST_LAVAPIPE", "false") == "true"
    using Lavapipe_jll
    ENV["VK_DRIVER_FILES"] = Lavapipe_jll.lvp_icd
    ENV["VK_ICD_FILENAMES"] = Lavapipe_jll.lvp_icd
end
