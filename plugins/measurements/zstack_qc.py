"""Z-Stack QC plugin — saturation, SNR, bleaching, and focus checks."""

from typing import Callable

import numpy as np

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.table_data import TableData
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port
from core.parameters import FloatParameter, IntParameter


def _robust_sigma(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    return 1.4826 * mad + 1e-6


def _run_qc(stack, saturation_value, warn_saturated, warn_snr, warn_bleaching):
    """Run all QC checks on a (Z, H, W) stack. Returns metrics dict."""
    if stack.ndim == 3 and stack.shape[0] > 0:
        proj = np.max(stack, axis=0)
    else:
        proj = stack

    x = proj.astype(np.float32).ravel()

    # Saturation
    sat = float(np.sum(proj >= saturation_value)) / float(proj.size) * 100.0 if proj.size else 0.0

    # Dynamic range
    dr = max(0.0, float(np.percentile(x, 99)) - float(np.percentile(x, 1))) if x.size else 0.0

    # SNR
    if x.size > 0:
        signal = float(np.percentile(x, 95))
        noise_px = x[x <= np.percentile(x, 25)]
        if noise_px.size < 100:
            noise_px = x
        noise = _robust_sigma(noise_px)
        snr = signal / noise if noise > 0 else 0.0
    else:
        snr = 0.0

    # Black pixels
    black = float(np.sum(proj == 0)) / float(proj.size) * 100.0 if proj.size else 0.0

    # Bleaching
    if stack.ndim == 3 and stack.shape[0] >= 2:
        nc = min(3, stack.shape[0] // 2)
        nc = max(1, nc)
        early, late = float(np.mean(stack[:nc])), float(np.mean(stack[-nc:]))
        bleach = (early - late) / early * 100.0 if early > 0 else 0.0
    else:
        bleach = 0.0

    # Slice CV
    if stack.ndim == 3 and stack.shape[0] >= 2:
        means = np.array([float(np.mean(stack[i])) for i in range(stack.shape[0])])
        avg = float(np.mean(means))
        cv = float(np.std(means, ddof=1) / avg) if avg > 0 else 0.0
    else:
        cv = 0.0

    # Outlier slices
    outliers = []
    if stack.ndim == 3 and stack.shape[0] >= 5:
        means = np.array([float(np.mean(stack[i])) for i in range(stack.shape[0])])
        med = float(np.median(means))
        rs = _robust_sigma(means)
        z_scores = np.abs(means - med) / rs
        outliers = np.where(z_scores > 3.0)[0].tolist()

    # Focus quality
    if stack.ndim == 3 and stack.shape[0] > 0:
        h, w = stack.shape[1], stack.shape[2]
        r = max(1, 50)
        r1, r2 = max(0, h // 2 - r), min(h, h // 2 + r)
        c1, c2 = max(0, w // 2 - r), min(w, w // 2 + r)
        fscores = np.array([float(np.std(stack[i, r1:r2, c1:c2].astype(np.float32)))
                            for i in range(stack.shape[0])])
        best, worst = float(np.max(fscores)), float(np.min(fscores))
        focus_drop = (best - worst) / best if best > 0 else 0.0
        best_z = int(np.argmax(fscores))
    else:
        focus_drop, best_z = 0.0, 0

    # Warnings
    warnings = []
    if sat > warn_saturated:
        warnings.append(f"Saturation: {sat:.2f}% (>{warn_saturated}%)")
    if snr < warn_snr:
        warnings.append(f"Low SNR: {snr:.1f} (<{warn_snr})")
    if bleach > warn_bleaching:
        warnings.append(f"Bleaching: {bleach:.1f}% (>{warn_bleaching}%)")
    if outliers:
        warnings.append(f"Outlier slices: {outliers}")

    return {
        "saturated_percent": sat,
        "dynamic_range": dr,
        "snr": snr,
        "black_percent": black,
        "bleaching_percent": bleach,
        "slice_cv": cv,
        "outlier_slices": outliers,
        "focus_drop": focus_drop,
        "focus_best_z": best_z,
        "n_slices": stack.shape[0] if stack.ndim == 3 else 1,
        "warnings": warnings,
        "passed": len(warnings) == 0,
    }


class ZStackQC(BasePlugin):
    """Run quality-control checks on a Z-stack.

    Measures saturation, SNR, dynamic range, bleaching, focus quality,
    and per-slice consistency.  Passes the image through unchanged and
    emits a TableData row with all metrics plus warnings.
    """

    name = "Z-Stack QC"
    category = "Measurements"
    description = "Saturation, SNR, bleaching, and focus quality checks"
    help_text = (
        "Runs quality-control checks on a Z-stack: saturation percentage, "
        "signal-to-noise ratio, dynamic range, photobleaching, focus "
        "variation, and per-slice consistency. Emits warnings when metrics "
        "exceed configurable thresholds. The image passes through unchanged."
    )
    icon = None

    ports: list[Port] = [
        InputPort("image_in", ImageContainer, label="Image In"),
        OutputPort("image_out", ImageContainer, label="Image Out"),
        OutputPort("measurements", TableData, label="QC Results"),
    ]

    parameters = [
        IntParameter(
            name="saturation_value",
            label="Saturation Value",
            default=65535,
            min_value=1,
            max_value=65535,
        ),
        FloatParameter(
            name="warn_saturated",
            label="Warn Saturated %",
            default=0.1,
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            decimals=2,
        ),
        FloatParameter(
            name="warn_snr",
            label="Warn SNR Below",
            default=3.0,
            min_value=0.0,
            max_value=100.0,
            step=0.5,
            decimals=1,
        ),
        FloatParameter(
            name="warn_bleaching",
            label="Warn Bleaching %",
            default=20.0,
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            decimals=1,
        ),
    ]

    def process(
        self,
        image: ImageContainer,
        progress_callback: Callable[[float], None],
    ) -> ImageContainer:
        return image

    def process_ports(
        self,
        inputs: dict[str, PipelineData],
        progress_callback: Callable[[float], None],
    ) -> dict[str, PipelineData]:
        image: ImageContainer = inputs.get("image_in")
        if image is None:
            raise RuntimeError("No image provided to Z-Stack QC")

        progress_callback(0.1)

        data = image.data
        if data.ndim == 4:
            stack = data[..., 0]
        elif data.ndim == 3:
            stack = data
        else:
            stack = data[np.newaxis, ...]

        progress_callback(0.2)

        qc_result = _run_qc(
            stack,
            int(self.get_parameter("saturation_value")),
            float(self.get_parameter("warn_saturated")),
            float(self.get_parameter("warn_snr")),
            float(self.get_parameter("warn_bleaching")),
        )

        progress_callback(0.8)

        row: dict = {}
        source = image.metadata.source_path
        row["filename"] = str(source.name) if source else ""
        for key, value in qc_result.items():
            if key == "warnings":
                row["warnings"] = "; ".join(value) if value else ""
            elif key == "outlier_slices":
                row["outlier_slices"] = str(value)
            else:
                row[key] = value

        table = TableData()
        table.add_row(row)

        progress_callback(1.0)
        return {"image_out": image, "measurements": table}

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
