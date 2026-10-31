# ============================================================================
# CoatedDiffuseTransmissionMaterial - Layered: dielectric coating + diffuse transmission
# ============================================================================
# Extension of CoatedDiffuseMaterial that uses DiffuseTransmissionBxDF as the
# bottom layer instead of DiffuseBxDF. This creates a material with a glossy
# dielectric coating over a diffuse substrate that both reflects and transmits.
#
# Ideal for: leaves, thin fabric, paper with a glossy surface coating.
#
# In pbrt-v4 terms: LayeredBxDF<DielectricBxDF, DiffuseTransmissionBxDF, true>

# CoatedDiffuseTransmissionMaterial struct, constructors, and alias
# are now in HikariBase. This file provides BSDF evaluation code.

is_emissive(::CoatedDiffuseTransmissionMaterial) = false
