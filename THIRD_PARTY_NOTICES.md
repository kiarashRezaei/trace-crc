# Third-Party Notices

This repository includes software developed by third parties. The project-level
MIT License does not replace their copyright notices.

## Learn then Test (LTT)

The files in `conformal/core/` are derived from the `core/` directory of:

- Project: Learn then Test
- Authors: Anastasios N. Angelopoulos and contributors
- Repository: <https://github.com/aangelopoulos/ltt>
- License: MIT
- Copyright: Copyright (c) 2021 Anastasios Angelopoulos

The vendored files were compared with upstream revision
`3ad7a64adad9f356e29f40e73935a6114f896396` while preparing this release.
Local changes include package-relative imports, optional handling of the
`confseq` dependency, numerical-stability safeguards, and cache-key fixes.

The complete upstream permission and warranty notice is retained in
[`conformal/core/LICENSE`](conformal/core/LICENSE).

When using the LTT implementation or methodology, please also cite:

> Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candes, Michael I.
> Jordan, and Lihua Lei. "Learn then Test: Calibrating Predictive Algorithms
> to Achieve Risk Control." *The Annals of Applied Statistics*, 19(2):
> 1641-1662, 2025. <https://doi.org/10.1214/24-AOAS1998>
