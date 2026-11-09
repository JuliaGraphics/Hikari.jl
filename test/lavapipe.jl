# Opt in before Hikari/Mantle creates the first Vulkan instance. Merely adding
# this driver to the search path can still pick a runner's hardware driver.
if get(ENV, "HIKARI_TEST_LAVAPIPE", "false") == "true"
    using Lavapipe_jll
    ENV["VK_DRIVER_FILES"] = Lavapipe_jll.lvp_icd
    ENV["VK_ICD_FILENAMES"] = Lavapipe_jll.lvp_icd
    # Lavapipe_jll is built without `memfd_create`, so Mesa backs all device memory with
    # files in `$XDG_RUNTIME_DIR` (`mesa-shared-*`). That is a tmpfs of about a tenth of RAM
    # (3.1 GB on a 32 GB desktop), while the driver reports all of RAM as its heap: a write
    # past the full tmpfs is a SIGBUS in the shader, and one 1200x360 render's state is
    # 2.2 GB. `/dev/shm` holds half of RAM. Remove this once Lavapipe_jll 26.2.4, which
    # creates that memory with memfd, is released.
    isdir("/dev/shm") && (ENV["XDG_RUNTIME_DIR"] = "/dev/shm")
end
