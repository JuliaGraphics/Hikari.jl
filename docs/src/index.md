```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Hikari.jl"
  tagline: Spectral path tracing in Julia, on Vulkan and Metal GPUs
  image:
    src: https://github.com/JuliaGraphics/Hikari.jl/releases/download/assets/protein_glass.png
    alt: A protein rendered in glass with Hikari
  actions:
    - theme: brand
      text: Get Started
      link: /get_started
    - theme: alt
      text: Materials
      link: /materials
    - theme: alt
      text: View on GitHub
      link: https://github.com/JuliaGraphics/Hikari.jl

features:
  - title: Spectral wavefront path tracing
    details: pbrt-v4's volumetric path tracer with hero-wavelength spectral transport, run as a wavefront of GPU kernels that Lava compiles from Julia, with hardware ray tracing where the GPU has it.
    link: /get_started
  - title: Volumes and clouds
    details: Delta tracking with null scattering for fog, smoke and clouds, including sparse NanoVDB volumes.
  - title: Blender's materials and node graphs
    details: Principled, Glass, Metallic and the other shader nodes, with Blender's names and defaults, driven by images or procedural node graphs.
    link: /materials
  - title: Lights
    details: Point, spot, directional, area and environment lights, with BVH light sampling for scenes with many emitters.
    link: /lights
  - title: Checked against pbrt-v4
    details: The test suite renders a pbrt-v4 reference set and compares every scene with pbrt's own render.
    link: /comparison
  - title: Denoising and Makie
    details: OIDN denoising with albedo and normal guides. RayMakie renders any Makie scene with Hikari.
    link: /postprocessing
---
```

```@raw html
<style>
/* The hero image is a square render, not a logo on a transparent background:
   at VitePress's full 320px it rises out of the hero and under the navigation bar. */
@media (min-width: 960px) {
  .VPHero.has-image .image-src { max-width: 256px; max-height: 256px; }
}
.VPHero.has-image .image-src { border-radius: 12px; }

/* The gallery: the BOMEX clouds as a banner, the rest in columns at their own aspect ratio. */
.hikari-gallery { max-width: 1152px; margin: 64px auto 0; padding: 0 24px; }
.hikari-gallery h2 { border-top: none; margin: 0 0 8px; padding-top: 0; }
.hikari-gallery > p { margin: 0 0 24px; color: var(--vp-c-text-2); }
.hikari-gallery figure { margin: 0 0 16px; break-inside: avoid; }
.hikari-gallery img { width: 100%; border-radius: 10px; display: block; }
.hikari-gallery figcaption { font-size: 14px; color: var(--vp-c-text-2); margin-top: 6px; }
.hikari-gallery .columns { columns: 3 280px; column-gap: 16px; }
</style>
```

```@raw html
<div class="hikari-gallery">
<h2>Gallery</h2>
<p>Rendered with Hikari through RayMakie. The scenes are in
<a href="https://github.com/SimonDanisch/RayDemo">SimonDanisch/RayDemo</a>; select a picture for the full render.</p>
<figure>
  <a href="https://github.com/JuliaGraphics/Hikari.jl/releases/download/assets/cloud_bomex.png"><img src="/gallery/cloud_bomex-1440.jpg" alt="A field of cumulus clouds" loading="lazy"></a>
  <figcaption>BOMEX large-eddy simulation cloud field, volumetric path tracing</figcaption>
</figure>
<div class="columns">
<figure>
  <a href="https://github.com/JuliaGraphics/Hikari.jl/releases/download/assets/terrain.png"><img src="/gallery/terrain-720.jpg" alt="A mountain terrain with clouds" loading="lazy"></a>
  <figcaption>Terrain with volumetric clouds</figcaption>
</figure>
<figure>
  <a href="https://github.com/JuliaGraphics/Hikari.jl/releases/download/assets/black_hole.png"><img src="/gallery/black_hole-720.jpg" alt="A black hole and its accretion disk" loading="lazy"></a>
  <figcaption>Black hole with gravitational lensing</figcaption>
</figure>
<figure>
  <a href="https://github.com/JuliaGraphics/Hikari.jl/releases/download/assets/protein_glass.png"><img src="/gallery/protein_glass-720.jpg" alt="A protein in glass" loading="lazy"></a>
  <figcaption>Protein structure in glass</figcaption>
</figure>
<figure>
  <a href="https://github.com/JuliaGraphics/Hikari.jl/releases/download/assets/cms_detector.png"><img src="/gallery/cms_detector-720.jpg" alt="The CMS particle detector" loading="lazy"></a>
  <figcaption>CMS particle detector, from <a href="https://github.com/JuliaHEP/Geant4.jl">Geant4.jl</a></figcaption>
</figure>
<figure>
  <a href="https://github.com/JuliaGraphics/Hikari.jl/releases/download/assets/spacecraft.png"><img src="/gallery/spacecraft-720.jpg" alt="The HL-20 spacecraft" loading="lazy"></a>
  <figcaption>HL-20 spacecraft with a volumetric exhaust</figcaption>
</figure>
<figure>
  <a href="https://github.com/JuliaGraphics/Hikari.jl/releases/download/assets/protein_gold.png"><img src="/gallery/protein_gold-720.jpg" alt="A protein in gold" loading="lazy"></a>
  <figcaption>Protein structure in gold</figcaption>
</figure>
<figure>
  <a href="https://github.com/JuliaGraphics/Hikari.jl/releases/download/assets/plants.png"><img src="/gallery/plants-720.jpg" alt="An oil palm" loading="lazy"></a>
  <figcaption>Oil palm, from <a href="https://github.com/VEZY/PlantGeom.jl">PlantGeom.jl</a></figcaption>
</figure>
<figure>
  <a href="https://github.com/JuliaGraphics/Hikari.jl/releases/download/assets/bunny_cloud.png"><img src="/gallery/bunny_cloud-720.jpg" alt="The Stanford bunny as a cloud" loading="lazy"></a>
  <figcaption>Stanford bunny cloud, a NanoVDB volume</figcaption>
</figure>
</div>
</div>
```
