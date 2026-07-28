# CSI Dataset

The full CSI dataset is not tracked by Git because it is approximately 195 MB
and exceeds GitHub's regular 100 MB per-file limit.

For full diffusion-inference reproduction, place the file here:

```text
data/csi/CSI_Dataset_1000_diverse.pt
```

Expected properties:

```text
type:  PyTorch float32 tensor
shape: (1000, 100, 2, 16, 16)
```

Expected SHA-256:

```text
cb19854166b662e4badcbc1f5374f4a424c93013ad5147d3e5bed078236a3d0c
```

Verify on macOS:

```bash
shasum -a 256 data/csi/CSI_Dataset_1000_diverse.pt
```

Verify on Linux:

```bash
sha256sum data/csi/CSI_Dataset_1000_diverse.pt
```

The compact-cache workflow does not require this file. Add a permanent release
or archival download URL here before making the repository public.

