# LTT Core

This directory is a modified vendored copy of the `core/` directory from
[aangelopoulos/ltt](https://github.com/aangelopoulos/ltt).

The upstream code is distributed under the MIT License:

```text
Copyright (c) 2021 Anastasios Angelopoulos
```

The complete upstream license is retained in [`LICENSE`](LICENSE). During
preparation of this release, the files were compared with upstream revision
`3ad7a64adad9f356e29f40e73935a6114f896396`.

Local changes provide package-relative imports, optional handling of the
`confseq` dependency, numerical-stability safeguards, and cache-key fixes.
See [`../../THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) for full
provenance and the recommended academic citation.
