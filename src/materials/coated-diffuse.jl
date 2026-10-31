# ============================================================================
# CoatedDiffuseMaterial - Evaluation code for layered dielectric + diffuse
# ============================================================================
# CoatedDiffuseMaterial struct and constructors are now in HikariBase.
# This file provides BSDF evaluation, shading, GPU support, etc.

# Mark as non-emissive
is_emissive(::CoatedDiffuseMaterial) = false
