```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: "Hikari.jl"
  tagline: Physically-based ray tracing on CPU & GPU
  image:
    src: https://github.com/JuliaGraphics/Hikari.jl/releases/download/assets/protein_glass.png
    alt: A protein rendered in glass with Hikari
  actions:
    - theme: brand
      text: Get Started
      link: /get_started
    - theme: alt
      text: View on Github
      link: https://github.com/JuliaGraphics/Hikari.jl
    - theme: alt
      text: API
      link: /api
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
</style>
```
