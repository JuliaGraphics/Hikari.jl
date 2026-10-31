# ============================================================================
# DiffuseTransmissionMaterial - Evaluation code for diffuse reflection + transmission
# ============================================================================
# DiffuseTransmissionMaterial struct, constructors, and DiffuseTransmission alias
# are now in HikariBase. This file provides BSDF evaluation code.

# Mark as non-emissive
is_emissive(::DiffuseTransmissionMaterial) = false
