# Load only in the pinned original checkout, after applying original-host-compat.patch.
# The old host selector asks about the removed RT-pipeline API. This comparison
# deliberately uses hardware ray queries for every Hikari implementation.
using Hikari, Mantle
@assert strip(read(Cmd(["git", "-C", pkgdir(Hikari), "rev-parse", "HEAD"]), String)) ==
    "d24058b103fb2c47e7709e7fd91c75aea4851a0b"
@eval Mantle supports_rt_pipeline(_) = false
