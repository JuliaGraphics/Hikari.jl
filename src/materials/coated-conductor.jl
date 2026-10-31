# ============================================================================
# CoatedConductorMaterial - Evaluation code for layered dielectric + conductor
# ============================================================================
# CoatedConductorMaterial struct and constructors are now in HikariBase.
# This file provides BSDF evaluation, shading, GPU support, etc.

# Mark as non-emissive
is_emissive(::CoatedConductorMaterial) = false
