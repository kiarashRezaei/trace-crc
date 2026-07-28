# Pretrained DiU Checkpoint

The pretrained diffusion checkpoint is not tracked by Git. For full
diffusion-inference reproduction, place it here:

```text
DiU/models/models_900.pt
```

Expected size:

```text
23,615,014 bytes
```

Expected SHA-256:

```text
800227c94359977bd80c4cef33a418fb06eb1ef5a0bddf52e5a7b73ba2e5f37b
```

Verify on macOS:

```bash
shasum -a 256 DiU/models/models_900.pt
```

Verify on Linux:

```bash
sha256sum DiU/models/models_900.pt
```

The compact-cache workflow does not require this checkpoint. Add a permanent
release or archival download URL here before making the repository public.

