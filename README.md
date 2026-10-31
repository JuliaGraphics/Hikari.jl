# Trace.jl
[![][docs-dev-img]][docs-dev-url]

Physically-based ray tracing on CPU

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://pxl-th.github.io/Trace.jl/dev/

<img src="./docs/src/assets/shadows-sppm-1024x1024_mio.png">

Output from `/scenes/shadows.jl`.

## Install

```
] add https://github.com/pxl-th/Trace.jl.git
```

## Examples

All of the examples are available in `scenes/` directory.

Example of running caustic glass scene (note that it supports multithreading).

```bash
julia -t 4 --project=. scenes/caustic_glass.jl
```

### Caustic simulation

Caustic effect using Stochastic Progressive Photon Mapping on a glass block.

|Image|Video|
|:-:|:-:|
|<img src="https://github.com/user-attachments/assets/f7d8cd98-1dfa-4bb2-9bb4-3a48e3556f1a" width=400>|<img src="https://i.ytimg.com/vi_webp/87NlMA3Vwvs/maxresdefault.webp" width="720">|
|100 iterations, ray depth 8|25 iterations per frame, ray depth 5. [YouTube](https://www.youtube.com/watch?v=87NlMA3Vwvs)|
