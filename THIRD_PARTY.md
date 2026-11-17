# Third-party source

`third_party/gaussian-splatting` is the official GraphDECO repository at commit
`54c035f7834b564019656c3e3fcc3646292f727d` with three small integration changes:

1. Preserve COLMAP principal points and use an off-axis projection matrix.
2. Propagate those principal points through camera construction.
3. Prefer this pipeline's explicit `sparse/0/test.txt` split over LLFF holdout logic.

The upstream repository and its submodules retain their original licenses.
