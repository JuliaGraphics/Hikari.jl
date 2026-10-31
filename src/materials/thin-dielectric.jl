# ============================================================================
# ThinDielectricMaterial - Evaluation code for thin dielectric surfaces
# ============================================================================
# ThinDielectricMaterial struct, keyword constructor, and ThinDielectric alias
# are now in HikariBase. This file provides BSDF evaluation code.

# Mark as non-emissive
is_emissive(::ThinDielectricMaterial) = false
