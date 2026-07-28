from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm


def get_best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_tf_memory_growth() -> None:
    import tensorflow as tf

    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass


def generate_cdl_dataset(
    num_samples: int,
    T: int = 40,
    n_sc_total: int = 300,
    n_sc_used: int = 16,
    n_tx: int = 16,
    fc: float = 28e9,
    subcarrier_spacing: float = 30e3,
    model_set: tuple[str, ...] = ("A", "B", "C", "D", "E"),
    delay_spread_range: tuple[float, float] = (50e-9, 400e-9),
    speed_kmh_range: tuple[float, float] = (30.0, 120.0),
    seed: int = 0,
) -> np.ndarray:
    """
    Returns CSI data with shape [num_samples, T, 2, n_tx, n_sc_used].
    The channel dimension is [real, imag].
    """
    import tensorflow as tf

    try:
        from sionna.phy.channel.tr38901 import CDL, PanelArray
        from sionna.phy.channel.utils import (
            cir_to_ofdm_channel,
            subcarrier_frequencies,
        )
    except Exception:
        from sionna.channel.tr38901 import CDL, PanelArray
        from sionna.channel.utils import cir_to_ofdm_channel, subcarrier_frequencies

    configure_tf_memory_growth()
    rng = np.random.default_rng(seed)

    bs_array = PanelArray(
        num_rows_per_panel=4,
        num_cols_per_panel=4,
        polarization="single",
        polarization_type="V",
        antenna_pattern="38.901",
        carrier_frequency=fc,
    )
    ut_array = PanelArray(
        num_rows_per_panel=1,
        num_cols_per_panel=1,
        polarization="single",
        polarization_type="V",
        antenna_pattern="omni",
        carrier_frequency=fc,
    )

    sc_idx = np.linspace(0, n_sc_total - 1, n_sc_used, dtype=int)
    freqs = subcarrier_frequencies(n_sc_total, subcarrier_spacing)
    H = np.zeros((num_samples, T, n_tx, n_sc_used), dtype=np.complex64)

    for i in tqdm(range(num_samples), desc="Generating CSI", leave=False):
        model = rng.choice(model_set)
        delay_spread = rng.uniform(*delay_spread_range)
        v_ms = rng.uniform(*speed_kmh_range) / 3.6

        cdl = CDL(
            model=model,
            delay_spread=delay_spread,
            carrier_frequency=fc,
            ut_array=ut_array,
            bs_array=bs_array,
            direction="downlink",
            min_speed=v_ms,
            max_speed=v_ms,
        )

        a, tau = cdl(
            batch_size=1,
            num_time_steps=T,
            sampling_frequency=subcarrier_spacing,
        )
        h_freq = cir_to_ofdm_channel(freqs, a, tau, normalize=True)
        h = h_freq[0, 0, 0, 0, :, :, :]
        h = tf.transpose(h, [1, 0, 2])
        h = tf.gather(h, sc_idx, axis=-1)
        H[i] = h.numpy()

    return np.stack([H.real, H.imag], axis=2).astype(np.float32)


def load_dataset_file(dataset_path: str | Path) -> np.ndarray:
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    if dataset_path.suffix == ".npy":
        return np.load(dataset_path).astype(np.float32)

    if dataset_path.suffix == ".pt":
        obj = torch.load(dataset_path, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().astype(np.float32)
        if isinstance(obj, np.ndarray):
            return obj.astype(np.float32)
        if isinstance(obj, dict):
            for key in ("X", "x", "data", "csi", "dataset"):
                if key in obj:
                    value = obj[key]
                    if isinstance(value, torch.Tensor):
                        return value.cpu().numpy().astype(np.float32)
                    return np.asarray(value, dtype=np.float32)
            raise ValueError(f"Unsupported dataset structure in {dataset_path}")

    raise ValueError(f"Unsupported dataset suffix: {dataset_path.suffix}")


def load_dataset_files(
    dataset_paths: Iterable[str | Path],
    *,
    max_samples_per_file: int | None = None,
    max_samples_total: int | None = None,
) -> np.ndarray:
    arrays: list[np.ndarray] = []

    for dataset_path in dataset_paths:
        arr = load_dataset_file(dataset_path)
        if max_samples_per_file is not None:
            arr = arr[:max_samples_per_file]
        arrays.append(arr)

    if not arrays:
        raise ValueError("No dataset files were provided.")

    first_shape = arrays[0].shape[1:]
    for idx, arr in enumerate(arrays[1:], start=1):
        if arr.shape[1:] != first_shape:
            raise ValueError(
                f"Dataset shape mismatch at index {idx}: expected (*, {first_shape}), got {arr.shape}"
            )

    merged = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
    if max_samples_total is not None:
        merged = merged[:max_samples_total]
    return merged.astype(np.float32)


def load_or_generate_dataset(
    dataset_path: str | Path,
    *,
    num_samples: int = 16,
    T: int = 40,
    seed: int = 0,
    force_regenerate: bool = False,
) -> np.ndarray:
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if dataset_path.exists() and not force_regenerate:
        return load_dataset_file(dataset_path)

    X = generate_cdl_dataset(num_samples=num_samples, T=T, seed=seed)
    if dataset_path.suffix == ".pt":
        torch.save(torch.from_numpy(X), dataset_path)
    else:
        np.save(dataset_path, X)
    return X


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_diffusers():
    try:
        from diffusers import DDIMScheduler, UNet2DModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "diffusers is required for DiU inference. Install it in the active "
            "environment before running the checkpoint-loading cell."
        ) from exc
    return UNet2DModel, DDIMScheduler


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding,
        )

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c


class ConvLSTMPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        hidden_channels: int = 128,
        kernel_size: int = 3,
        final_activation: str | None = "relu",
    ):
        super().__init__()
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size=kernel_size)
        self.norm = nn.BatchNorm2d(hidden_channels)
        self.conv_out = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.final_activation = final_activation

    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.final_activation is None:
            return x
        if self.final_activation == "relu":
            return torch.relu(x)
        if self.final_activation == "tanh":
            return torch.tanh(x)
        if self.final_activation == "sigmoid":
            return torch.sigmoid(x)
        raise ValueError(f"Unsupported activation: {self.final_activation}")

    def forward(
        self,
        x_past: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _, H, W = x_past.shape
        h = torch.zeros(B, self.cell.hidden_channels, H, W, device=x_past.device)
        c = torch.zeros_like(h)

        for t in range(T):
            h, c = self.cell(x_past[:, t], h, c)

        h = self.norm(h)
        out = self.conv_out(h)
        out = self._apply_activation(out)
        return out.unsqueeze(1), h, c


def strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for key, value in state_dict.items():
        if key == "n_averaged":
            continue
        out[key[7:] if key.startswith("module.") else key] = value
    return out


def build_diu_models(
    config_path: str | Path,
    *,
    device: torch.device,
    image_size: int = 16,
    data_channels: int = 2,
):
    raw_config = load_yaml(config_path)
    diff_cfg = raw_config["model"]["Diffusion"]
    UNet2DModel, DDIMScheduler = require_diffusers()

    lstm = ConvLSTMPredictor(
        in_channels=data_channels,
        out_channels=diff_cfg["LSTM_out_channels"],
        hidden_channels=diff_cfg["LSTM_hidden_channels"],
        kernel_size=diff_cfg["LSTM_kernel_size"],
        final_activation=diff_cfg.get("final_activation"),
    ).to(device)

    unet = UNet2DModel(
        sample_size=image_size,
        in_channels=data_channels + diff_cfg["LSTM_out_channels"],
        out_channels=data_channels,
        layers_per_block=diff_cfg["Unet_layers_per_block"],
        block_out_channels=tuple(diff_cfg["Unet_block_out_channels"]),
        down_block_types=tuple(diff_cfg["down_block_types"]),
        up_block_types=tuple(diff_cfg["up_block_types"]),
        norm_num_groups=diff_cfg["Unet_norm_num_groups"],
    ).to(device)

    scheduler = DDIMScheduler(
        num_train_timesteps=diff_cfg["timesteps"],
        beta_schedule=diff_cfg["beta_schedule"],
        beta_start=diff_cfg["beta_start"],
        beta_end=diff_cfg["beta_end"],
        prediction_type=diff_cfg["prediction_type"],
    )
    return raw_config, lstm, unet, scheduler


def load_diu_checkpoint(
    checkpoint_path: str | Path,
    config_path: str | Path,
    *,
    device: torch.device | None = None,
    image_size: int = 16,
    data_channels: int = 2,
    use_ema: bool = True,
) -> dict[str, Any]:
    device = device or get_best_device()
    raw_config, lstm, unet, scheduler = build_diu_models(
        config_path,
        device=device,
        image_size=image_size,
        data_channels=data_channels,
    )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    lstm.load_state_dict(ckpt["LSTM_model"])

    if use_ema and "model_ema" in ckpt:
        unet.load_state_dict(strip_module_prefix(ckpt["model_ema"]), strict=True)
        model_source = "EMA"
    else:
        unet.load_state_dict(ckpt["model"], strict=True)
        model_source = "base"

    lstm.eval()
    unet.eval()

    return {
        "config": raw_config,
        "checkpoint": ckpt,
        "lstm": lstm,
        "unet": unet,
        "scheduler": scheduler,
        "normalization_state": ckpt.get("normalization_state", {}),
        "device": device,
        "model_source": model_source,
    }


def _normalization_method(state: dict[str, Any]) -> str:
    return str(state.get("normalize_method", "none")).lower()


def normalize_csi(x: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    method = _normalization_method(state)
    if method == "none" or not state:
        return x
    if method == "minmax":
        a = torch.as_tensor(state["a"], dtype=x.dtype, device=x.device)
        b = torch.as_tensor(state["b"], dtype=x.dtype, device=x.device)
        return 2.0 * (x - a) / (b - a + 1e-9) - 1.0
    if method == "zscore":
        mean = torch.as_tensor(state["a"], dtype=x.dtype, device=x.device)
        std = torch.as_tensor(state["b"], dtype=x.dtype, device=x.device)
        return (x - mean) / (std + 1e-9)
    raise ValueError(f"Unsupported normalization method: {method}")


def denormalize_csi(x: torch.Tensor, state: dict[str, Any]) -> torch.Tensor:
    method = _normalization_method(state)
    if method == "none" or not state:
        return x
    if method == "minmax":
        a = torch.as_tensor(state["a"], dtype=x.dtype, device=x.device)
        b = torch.as_tensor(state["b"], dtype=x.dtype, device=x.device)
        return (x + 1.0) * 0.5 * (b - a) + a
    if method == "zscore":
        mean = torch.as_tensor(state["a"], dtype=x.dtype, device=x.device)
        std = torch.as_tensor(state["b"], dtype=x.dtype, device=x.device)
        return x * std + mean
    raise ValueError(f"Unsupported normalization method: {method}")


def add_snr_noise(x: torch.Tensor, snr_db: float | torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(snr_db):
        snr_db = torch.tensor(float(snr_db), device=x.device, dtype=x.dtype)
    snr_db = snr_db.to(device=x.device, dtype=x.dtype)
    if snr_db.ndim == 0:
        snr_db = snr_db.repeat(x.shape[0])

    view_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    power = x.float().pow(2).mean(dim=tuple(range(1, x.ndim)), keepdim=True)
    snr_linear = 10.0 ** (snr_db.view(view_shape) / 10.0)
    noise_power = power / (snr_linear + 1e-9)
    noise = torch.randn_like(x) * torch.sqrt(noise_power)
    return x + noise


def predict_single_frame(
    past: torch.Tensor,
    *,
    lstm: nn.Module,
    unet: nn.Module,
    scheduler,
    sampling_steps: int = 20,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    deterministic: bool = True,
) -> torch.Tensor:
    device = past.device
    with torch.inference_mode():
        lstm_output, _, _ = lstm(past)
        cond = lstm_output.squeeze(1)
        B, _, H, W = cond.shape

        if deterministic:
            samples = torch.zeros((B, 2, H, W), device=device, dtype=past.dtype)
        else:
            samples = torch.randn((B, 2, H, W), device=device, dtype=past.dtype)

        scheduler.set_timesteps(sampling_steps, device=device)

        for t in scheduler.timesteps:
            if guidance_scale != 1.0:
                zeros = torch.zeros_like(cond)
                model_inputs = torch.cat(
                    [
                        torch.cat([samples, cond], dim=1),
                        torch.cat([samples, zeros], dim=1),
                    ],
                    dim=0,
                )
                t_batch = t.expand(model_inputs.shape[0])
                preds = unet(model_inputs, t_batch).sample
                pred_cond, pred_uncond = preds.chunk(2, dim=0)
                model_pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                model_input = torch.cat([samples, cond], dim=1)
                model_pred = unet(model_input, t).sample

            samples = scheduler.step(model_pred, t, samples, eta=eta).prev_sample

        return samples


def auto_regressive_inference(
    past: torch.Tensor,
    *,
    horizon: int,
    lstm: nn.Module,
    unet: nn.Module,
    scheduler,
    sampling_steps: int = 20,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    noise_level: float = 100.0,
    show_progress: bool = True,
) -> torch.Tensor:
    context = past.clone()
    outputs = []
    iterator = range(horizon)
    if show_progress:
        iterator = tqdm(iterator, desc="AR inference", leave=False)

    with torch.inference_mode():
        for _ in iterator:
            samples = predict_single_frame(
                context,
                lstm=lstm,
                unet=unet,
                scheduler=scheduler,
                sampling_steps=sampling_steps,
                eta=eta,
                guidance_scale=guidance_scale,
                deterministic=True,
            )
            outputs.append(samples)

            signal_power = torch.mean(samples.float() ** 2, dim=(-3, -2, -1), keepdim=True)
            snr_linear = 1.2 * 10.0 ** (noise_level / 10.0)
            noise_power = signal_power / snr_linear
            new_data = samples + torch.sqrt(noise_power) * torch.randn_like(samples)
            context = torch.cat([context, new_data.unsqueeze(1)], dim=1)

    return torch.stack(outputs, dim=1)


def compute_nmse_db(pred: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    assert pred.shape == target.shape
    num = torch.sum((pred - target) ** 2, dim=(0, 2, 3, 4))
    den = torch.sum(target ** 2, dim=(0, 2, 3, 4)) + 1e-9
    return (10.0 * torch.log10(num / den)).cpu().numpy()


def compute_frobenius_norm(
    csi: torch.Tensor | np.ndarray,
    *,
    channel_dim: int = -3,
) -> torch.Tensor | np.ndarray:
    """
    Compute the Frobenius norm of a CSI tensor.

    Supported inputs:
    - Real/imag representation with shape like [2, H, W] or [..., 2, H, W]
    - Native complex tensors/arrays with shape like [H, W] or [..., H, W]

    Args:
        csi: CSI tensor or array.
        channel_dim: Position of the real/imag channel for non-complex inputs.

    Returns:
        Frobenius norm. For batched input, returns one value per leading item.
    """
    if isinstance(csi, np.ndarray):
        if np.iscomplexobj(csi):
            return np.sqrt(np.sum(np.abs(csi) ** 2, axis=(-2, -1)))
        if csi.shape[channel_dim] != 2:
            raise ValueError(
                f"Expected real/imag CSI with channel size 2 at axis {channel_dim}, got shape {csi.shape}"
            )
        return np.sqrt(np.sum(csi**2, axis=(channel_dim, -2, -1)))

    if torch.is_complex(csi):
        return torch.sqrt(torch.sum(torch.abs(csi) ** 2, dim=(-2, -1)))
    if csi.shape[channel_dim] != 2:
        raise ValueError(
            f"Expected real/imag CSI with channel size 2 at axis {channel_dim}, got shape {tuple(csi.shape)}"
        )
    return torch.sqrt(torch.sum(csi**2, dim=(channel_dim, -2, -1)))


def run_inference_and_save(
    dataset: np.ndarray,
    bundle: dict[str, Any],
    *,
    output_dir: str | Path,
    T_in: int = 30,
    T_out: int = 10,
    inference_snr: float = 20.0,
    batch_size: int = 4,
    sampling_steps: int = 20,
    eta: float = 0.0,
    guidance_scale: float = 1.0,
    noise_level: float = 100.0,
) -> dict[str, Any]:
    if dataset.ndim != 5:
        raise ValueError(f"Expected dataset shape [N, T, 2, H, W], got {dataset.shape}")
    if dataset.shape[1] < T_in + T_out:
        raise ValueError(
            f"Dataset length {dataset.shape[1]} is too short for T_in={T_in}, T_out={T_out}"
        )

    device = bundle["device"]
    lstm = bundle["lstm"]
    unet = bundle["unet"]
    scheduler = bundle["scheduler"]
    norm_state = bundle["normalization_state"]
    raw_cfg = bundle["config"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = torch.from_numpy(dataset).float()
    past_all = data[:, :T_in]
    target_all = data[:, T_in : T_in + T_out]

    preds = []
    for start in tqdm(range(0, len(data), batch_size), desc="Batches", leave=False):
        end = min(start + batch_size, len(data))
        past = past_all[start:end].to(device)
        past_norm = normalize_csi(past, norm_state)
        past_noisy = add_snr_noise(past_norm, inference_snr)
        pred_norm = auto_regressive_inference(
            past_noisy,
            horizon=T_out,
            lstm=lstm,
            unet=unet,
            scheduler=scheduler,
            sampling_steps=sampling_steps,
            eta=eta,
            guidance_scale=guidance_scale,
            noise_level=noise_level,
            show_progress=False,
        )
        preds.append(denormalize_csi(pred_norm, norm_state).cpu())

    pred_all = torch.cat(preds, dim=0)
    nmse_db = compute_nmse_db(pred_all, target_all)

    pt_path = output_dir / "predicted_csi.pt"
    npy_path = output_dir / "predicted_csi.npy"
    meta_path = output_dir / "prediction_summary.yaml"

    torch.save(
        {
            "predicted_csi": pred_all,
            "target_csi": target_all,
            "input_csi": past_all,
            "nmse_db_per_step": nmse_db,
            "inference_snr_db": inference_snr,
            "sampling_steps": sampling_steps,
            "model_source": bundle["model_source"],
            "checkpoint_epoch": int(bundle["checkpoint"].get("epoch", -1)),
        },
        pt_path,
    )
    np.save(npy_path, pred_all.numpy())

    summary = {
        "dataset_shape": list(dataset.shape),
        "predicted_shape": list(pred_all.shape),
        "checkpoint_epoch": int(bundle["checkpoint"].get("epoch", -1)),
        "model_source": bundle["model_source"],
        "inference_snr_db": float(inference_snr),
        "sampling_steps": int(sampling_steps),
        "normalization_method": raw_cfg["dataset"]["normalization_method"],
        "nmse_db_per_step": [float(x) for x in nmse_db.tolist()],
        "predicted_csi_pt": str(pt_path),
        "predicted_csi_npy": str(npy_path),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    summary["summary_yaml"] = str(meta_path)
    return summary
