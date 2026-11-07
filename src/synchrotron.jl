# Synchrotron physics moved from PlasmaScope/src/field.jl.
# Derived from the GRMHDSnapshots/PlasmaScope model by Alexander Plavin;
# original MIT notice retained in third_party/GRMHDSnapshots-LICENSE.
# Shared headless model: no viewer, image generation or display transform.
using Unitful
using SpecialFunctions: gamma

# ── Local power-law synchrotron emission model (plain CGS: ne cm⁻³, B Gauss, ν Hz) ──
# Physical constants stripped to plain CGS Float64 (matching the downstream synchrotron code).
const MEC2 = ustrip(u"erg", Unitful.me*Unitful.c0^2)   # electron rest energy mₑc²
const MPME = ustrip(u"g",   Unitful.mp + Unitful.me)   # ion + electron mass

# Pluggable nonthermal electron-density model.
abstract type NeModel end
struct MagneticEtaNe{T} <: NeModel; η::T; end          # fraction η of B²/(8π), B in Gaussian CGS
struct RhoNe{T} <: NeModel; scale::T; end              # ne ∝ thermal number density ρ·ρunit/(mₚ+mₑ)

electron_density(m::MagneticEtaNe, B_gauss, ρ_code, ρunit, p, γmin) =
    m.η * B_gauss^2/(8π) * (p - 2)/((p - 1)*MEC2*γmin)
electron_density(m::RhoNe, B_gauss, ρ_code, ρunit, p, γmin) =
    m.scale * ρ_code * ρunit / MPME

# Power-law normalization N(γ)=Kγ^(-p) on [γmin,γmax]: K/ne = (p-1)/(γmin^(1-p) - γmax^(1-p)).
K(p, γmin, γmax) = (p - 1)/(γmin^(1 - p) - γmax^(1 - p))

# Ordered-field CGS emissivity coefficient c₅(p) (Rybicki–Lightman power-law synchrotron).
function c₅(p)
    e = 4.8032068e-10                    # electron charge [statC]
    me = ustrip(u"g", Unitful.me)
    c = ustrip(Float64, u"cm/s", Unitful.c0)   # Float64: c0 is exact-integer, c² would overflow Int64
    A = 2π*me*c/(3e)
    pref0 = √3*e^3/(4π*me*c^2)
    pref0/(p + 1)*A^(-(p - 1)/2)*gamma((3p - 1)/12)*gamma((3p + 19)/12)
end

# Local emissivity in the "maximum direction" (B_perp = |B_comoving|, isotropic electrons, φ=1).
_jν(Cj, ne, B_perp, p, ν) = Cj*ne*B_perp^((p + 1)/2)*ν^(-(p - 1)/2)
synchrotron_jν(ne, B_perp, p, ν, γmin, γmax) = _jν(c₅(p)*K(p, γmin, γmax), ne, B_perp, p, ν)

# Stokes-I power-law absorption, Pandya et al. (2016), equations 30/34;
# same expression as ipole's symphony/power_law_fits.c, in plain CGS.
function synchrotron_αν(ne, B_perp, p, ν, γmin, γmax)
    e = 4.8032068e-10
    me = ustrip(u"g", Unitful.me)
    c = ustrip(Float64, u"cm/s", Unitful.c0)
    νc = e * B_perp / (2π * me * c)
    ne * e^2 / (ν * me * c) * 3^((p+1)/2) * K(p, γmin, γmax) / 4 *
        gamma((3p+2)/12) * gamma((3p+22)/12) * (νc / ν)^((p+2)/2)
end

# Emission parameters carried by the field source; UI drives these.
@kwdef struct EmissionParams
    ne::NeModel = MagneticEtaNe(0.01)
    p = 2.5
    ν = 2.3e11
    γmin = 10.0
    γmax = 1e5
    σcut = 1.0
end


"""
    radiative_volume(path; mass_unit, black_hole_mass, resolution=192,
                     extent=15f0, normalization=nothing, emission_params=EmissionParams())

Prepare emissivity, absorption and fluid-velocity fields for a volume medium.
Loading a GRMHD snapshot requires loading the optional GRMHDSnapshots package.
Coordinates use Schwarzschild radii. No images or display transforms are made.
The simulation mass unit and black-hole mass must be supplied in CGS grams;
the adapter does not choose a particular astronomical object or flux calibration.
"""
function radiative_volume end

"""
    native_radiative_volume(path; mass_unit, black_hole_mass, fade=(11f0, 14f0),
                            normalization=nothing, emission_params=EmissionParams())

The fields of [`radiative_volume`](@ref) on the snapshot's own spherical grid
instead of a Cartesian resampling: emissivity, absorption and the fluid's
static-frame Cartesian velocity per native cell, `(φ, θ, r)`, radial shells up to
the first one at `fade[2]`, faded out from `fade[1]` on. The grid comes along as
the colatitude of every cell (`theta[r, θ]`), the log-uniform radial spacing and
the uniform azimuth spacing. Radii are in Schwarzschild radii.
"""
function native_radiative_volume end
function emissivity_grid end
function absorption_grid end
function emission_contribution_grid end
function image_contribution_grid end
