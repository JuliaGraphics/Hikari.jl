# Hikari vs pbrt-v4

Hikari's test suite renders every scene of a pbrt-v4 reference set and compares each render with pbrt-v4's. Two scores decide whether a scene passes:

- The **tile score** is a perceptual difference. It is the 95th percentile, over 16x16 pixel tiles, of the mean color distance in log space.
- The **energy ratio** is Hikari's total luminance divided by pbrt-v4's.

```@raw html
<p><a href="pbrt-comparison/gallery.html" target="_blank">Open the comparison gallery</a>
of the latest CI run. It lists every scene, worst first, and its header gives the run's samples
per pixel and tolerances. Click an image to switch between Hikari and pbrt-v4.</p>
```
