"""Colocalization analysis plugin — Manders, Pearson, Jaccard, ICQ, Costes, Object-3D metrics."""

from typing import Callable

import numpy as np
from skimage.filters import threshold_otsu

from core.plugin_base import BasePlugin
from core.image_container import ImageContainer, ImageType
from core.table_data import TableData
from core.pipeline_data import PipelineData
from core.ports import InputPort, OutputPort, Port
from core.parameters import BoolParameter, ChoiceParameter, FloatParameter, IntParameter


def _threshold_mask(img: np.ndarray, method: str, percentile: float) -> np.ndarray:
    """Create a binary mask via Otsu or percentile threshold."""
    x = img.astype(np.float32)
    p1, p99 = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    if (p99 - p1) < 10:
        return np.zeros_like(x, dtype=np.float32)

    if method == "percentile":
        nz = x[x > 0]
        if nz.size < 100:
            return np.zeros_like(x, dtype=np.float32)
        thr = float(np.percentile(nz, percentile))
    else:  # otsu
        flat = x.ravel()
        if flat.size > 100000:
            flat = flat[:: int(flat.size // 100000 + 1)]
        if flat.size < 100:
            return np.zeros_like(x, dtype=np.float32)
        thr = float(threshold_otsu(flat))

    return (x > thr).astype(np.float32)


def _threshold_value(img: np.ndarray, method: str, percentile: float) -> float:
    """Return the threshold value (used by ICQ to get the numeric threshold)."""
    x = img.astype(np.float32)
    p1, p99 = float(np.percentile(x, 1)), float(np.percentile(x, 99))
    if (p99 - p1) < 10:
        return float("inf")

    if method == "percentile":
        nz = x[x > 0]
        if nz.size < 100:
            return float("inf")
        return float(np.percentile(nz, percentile))

    flat = x.ravel()
    if flat.size > 100000:
        flat = flat[:: int(flat.size // 100000 + 1)]
    if flat.size < 100:
        return float("inf")
    return float(threshold_otsu(flat))


def _pearson_r(v1: np.ndarray, v2: np.ndarray) -> float:
    """Pearson correlation between two 1D float64 arrays."""
    if v1.size < 10:
        return 0.0
    d1, d2 = v1 - np.mean(v1), v2 - np.mean(v2)
    den = float(np.sqrt(np.sum(d1 ** 2) * np.sum(d2 ** 2)))
    return float(np.sum(d1 * d2)) / den if den > 0 else 0.0


class Colocalization(BasePlugin):
    """Compute colocalization metrics between two images.

    For Z-stacks, per-slice analysis is performed with weighted
    aggregation.  For single images, one slice of analysis is run.

    Core metrics: Manders M1/M2, Pearson r, Jaccard index.
    Optional: Li ICQ, Costes randomisation, object-based 3D, volumetric M1/M2.
    """

    name = "Colocalization"
    category = "Measurements"
    description = "Manders, Pearson, Jaccard, ICQ, Costes colocalization metrics"
    help_text = (
        "Computes colocalization metrics between two images. Core metrics: "
        "Manders M1/M2 (fraction of signal overlapping), Pearson r (linear "
        "correlation), Jaccard index (overlap area ratio). Optional: "
        "Volumetric M1/M2 (intensity-weighted across full stack), Li ICQ "
        "(intensity correlation quotient, -0.5 to +0.5), Costes randomisation "
        "(statistical significance via block scrambling), Object-based 3D "
        "(connected-component overlap counting). Enable optional metrics "
        "via their checkboxes."
    )
    icon = None

    ports: list[Port] = [
        InputPort("image1_in", ImageContainer, label="Image 1"),
        InputPort("image2_in", ImageContainer, label="Image 2"),
        OutputPort("measurements", TableData, label="Measurements"),
    ]

    parameters = [
        ChoiceParameter(
            name="threshold_method",
            label="Threshold Method",
            choices=["Otsu", "Percentile"],
            default="Otsu",
        ),
        FloatParameter(
            name="percentile",
            label="Percentile",
            default=90.0,
            min_value=50.0,
            max_value=99.9,
            step=1.0,
            decimals=1,
        ),
        ChoiceParameter(
            name="aggregate_mode",
            label="Aggregate Mode",
            choices=["Weighted", "Equal"],
            default="Weighted",
        ),
        BoolParameter(
            name="compute_volumetric",
            label="Volumetric M1/M2",
            default=False,
        ),
        BoolParameter(
            name="compute_icq",
            label="Li ICQ",
            default=False,
        ),
        BoolParameter(
            name="compute_costes",
            label="Costes Randomisation",
            default=False,
        ),
        IntParameter(
            name="costes_n_scrambles",
            label="Costes Scrambles",
            default=200,
            min_value=50,
            max_value=1000,
        ),
        IntParameter(
            name="costes_block_size",
            label="Costes Block Size",
            default=3,
            min_value=2,
            max_value=10,
        ),
        BoolParameter(
            name="compute_object_3d",
            label="Object-based 3D",
            default=False,
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
        image1: ImageContainer = inputs.get("image1_in")
        image2: ImageContainer = inputs.get("image2_in")
        if image1 is None or image2 is None:
            raise RuntimeError("Colocalization requires two image inputs")

        progress_callback(0.05)

        method = self.get_parameter("threshold_method").lower()
        pct = float(self.get_parameter("percentile"))
        weighted = self.get_parameter("aggregate_mode").lower() == "weighted"
        do_volumetric = self.get_parameter("compute_volumetric")
        do_icq = self.get_parameter("compute_icq")
        do_costes = self.get_parameter("compute_costes")
        do_object_3d = self.get_parameter("compute_object_3d")

        stack1 = self._to_gray_stack(image1.data)
        stack2 = self._to_gray_stack(image2.data)
        n = min(stack1.shape[0], stack2.shape[0])

        # --- Core per-slice loop (Manders, Pearson, Jaccard) ---
        weights, M1s, M2s, Rs, Js = [], [], [], [], []
        total_overlap = total_ch1 = total_ch2 = 0

        # Volumetric accumulators
        vol_ch1_total = vol_ch2_total = vol_ch1_coloc = vol_ch2_coloc = 0.0

        # ICQ accumulators
        icq_vals: list[float] = []
        icq_weights: list[float] = []

        # Costes accumulators
        costes_results: list[dict] = []

        # Object-3D: collect mask stacks
        mask_stack_1 = np.zeros((n,) + stack1.shape[1:], dtype=bool) if do_object_3d else None
        mask_stack_2 = np.zeros((n,) + stack2.shape[1:], dtype=bool) if do_object_3d else None

        # Determine progress allocation
        core_weight = 0.6
        extra_weight = 0.2  # for costes/object-3d after main loop

        for z in range(n):
            img1 = stack1[z].astype(np.float32)
            img2 = stack2[z].astype(np.float32)
            mask1 = _threshold_mask(img1, method, pct)
            mask2 = _threshold_mask(img2, method, pct)

            m1, m2 = mask1 > 0, mask2 > 0
            ov = m1 & m2
            un = m1 | m2
            ch1_px, ch2_px, ov_px, un_px = (
                int(np.sum(m1)), int(np.sum(m2)),
                int(np.sum(ov)), int(np.sum(un)),
            )

            # Store masks for object-3D
            if do_object_3d:
                mask_stack_1[z] = m1
                mask_stack_2[z] = m2

            # Volumetric accumulators
            if do_volumetric or True:  # always accumulate for volumetric columns
                if ch1_px > 0:
                    vol_ch1_total += float(np.sum(img1[m1]))
                if ch2_px > 0:
                    vol_ch2_total += float(np.sum(img2[m2]))
                if ov_px > 0:
                    vol_ch1_coloc += float(np.sum(img1[ov]))
                    vol_ch2_coloc += float(np.sum(img2[ov]))

            if ch1_px > 0 and ch2_px > 0:
                # Manders
                ch1_t = float(np.sum(img1[m1]))
                ch2_t = float(np.sum(img2[m2]))
                mM1 = float(np.sum(img1[ov])) / ch1_t if ch1_t > 0 else 0.0
                mM2 = float(np.sum(img2[ov])) / ch2_t if ch2_t > 0 else 0.0

                # Pearson on union
                v1, v2 = img1[un].astype(np.float64), img2[un].astype(np.float64)
                r = _pearson_r(v1, v2)

                j = (ov_px / un_px * 100.0) if un_px > 0 else 0.0

                w = float(un_px) if weighted and un_px > 0 else 1.0
                weights.append(w)
                M1s.append(mM1)
                M2s.append(mM2)
                Rs.append(r)
                Js.append(j)

            total_overlap += ov_px
            total_ch1 += ch1_px
            total_ch2 += ch2_px

            # ICQ per slice
            if do_icq:
                t1 = _threshold_value(img1, method, pct)
                t2 = _threshold_value(img2, method, pct)
                if np.isfinite(t1) and np.isfinite(t2):
                    icq_mask = (img1 > t1) | (img2 > t2)
                    npx = int(icq_mask.sum())
                    if npx > 0:
                        iv1 = img1[icq_mask].astype(np.float64)
                        iv2 = img2[icq_mask].astype(np.float64)
                        prod = (iv1 - float(iv1.mean())) * (iv2 - float(iv2.mean()))
                        icq = (float((prod > 0).sum()) / float(npx)) - 0.5
                        icq_w = float(npx) if weighted else 1.0
                        icq_vals.append(icq)
                        icq_weights.append(icq_w)

            progress_callback(0.05 + core_weight * (z + 1) / n)

        # --- Costes randomisation (after main loop for cleaner progress) ---
        if do_costes:
            n_scrambles = int(self.get_parameter("costes_n_scrambles"))
            block_size = int(self.get_parameter("costes_block_size"))
            seed = 42
            rng = np.random.default_rng(seed)

            for z in range(n):
                img1 = stack1[z].astype(np.float32)
                img2 = stack2[z].astype(np.float32)

                # Observed Pearson r
                un = (_threshold_mask(img1, method, pct) > 0) | (_threshold_mask(img2, method, pct) > 0)
                if int(np.sum(un)) < 10:
                    continue

                v1_obs = img1[un].astype(np.float64)
                v2_obs = img2[un].astype(np.float64)
                observed_r = _pearson_r(v1_obs, v2_obs)

                # Block scramble channel 2
                h, w = img2.shape[:2]
                blocks = [(y, x) for y in range(0, h, block_size) for x in range(0, w, block_size)]

                scrambled_rs = np.empty(n_scrambles, dtype=np.float64)
                for si in range(n_scrambles):
                    idx = np.arange(len(blocks))
                    rng.shuffle(idx)
                    temp = np.zeros_like(img2)
                    for new_i, old_i in enumerate(idx):
                        sy, sx = blocks[new_i]
                        oy, ox = blocks[old_i]
                        ey = min(sy + block_size, h)
                        ex = min(sx + block_size, w)
                        oey = min(oy + block_size, h)
                        oex = min(ox + block_size, w)
                        bh = min(ey - sy, oey - oy)
                        bw = min(ex - sx, oex - ox)
                        temp[sy:sy + bh, sx:sx + bw] = img2[oy:oy + bh, ox:ox + bw]

                    v2_scr = temp[un].astype(np.float64)
                    scrambled_rs[si] = _pearson_r(v1_obs, v2_scr)

                p_value = float(np.mean(scrambled_rs < observed_r))
                costes_results.append({
                    "observed_r": float(observed_r),
                    "mean_scrambled_r": float(np.mean(scrambled_rs)),
                    "p_value": float(p_value),
                    "significant": bool(p_value >= 0.95),
                })

                progress_callback(0.05 + core_weight + extra_weight * (z + 1) / n)

        # --- Object-based 3D ---
        obj_3d_row = {}
        if do_object_3d:
            try:
                from scipy.ndimage import label as ndimage_label
            except ImportError:
                ndimage_label = None

            if ndimage_label is not None and mask_stack_1 is not None:
                m1_3d = mask_stack_1
                m2_3d = mask_stack_2
                ov_3d = m1_3d & m2_3d
                un_3d = m1_3d | m2_3d

                obj_3d_row["voxels_ch1"] = int(m1_3d.sum())
                obj_3d_row["voxels_ch2"] = int(m2_3d.sum())
                obj_3d_row["voxels_overlap"] = int(ov_3d.sum())
                obj_3d_row["voxels_union"] = int(un_3d.sum())
                obj_3d_row["jaccard_voxels"] = (
                    float(ov_3d.sum()) / float(un_3d.sum()) if int(un_3d.sum()) > 0 else 0.0
                )

                lab1, n1 = ndimage_label(m1_3d)
                lab2, n2 = ndimage_label(m2_3d)
                obj_3d_row["n_objects_ch1"] = int(n1)
                obj_3d_row["n_objects_ch2"] = int(n2)

                ch1_overlap = 0
                for i in range(1, n1 + 1):
                    if np.any((lab1 == i) & m2_3d):
                        ch1_overlap += 1
                ch2_overlap = 0
                for j in range(1, n2 + 1):
                    if np.any((lab2 == j) & m1_3d):
                        ch2_overlap += 1

                obj_3d_row["n_objects_ch1_overlapping_ch2"] = ch1_overlap
                obj_3d_row["n_objects_ch2_overlapping_ch1"] = ch2_overlap

        # --- Aggregate ---
        def _wmean(vals, wts):
            if not vals:
                return 0.0
            w = np.asarray(wts, dtype=np.float64)
            v = np.asarray(vals, dtype=np.float64)
            s = float(np.sum(w))
            return float(np.sum(w * v) / s) if s > 0 else float(np.mean(v))

        def _wstd(vals, wts):
            if len(vals) < 2:
                return 0.0
            w = np.asarray(wts, dtype=np.float64)
            v = np.asarray(vals, dtype=np.float64)
            mu = _wmean(vals, wts)
            s = float(np.sum(w))
            if s <= 0:
                return float(np.std(v, ddof=1))
            denom = s - float(np.sum(w ** 2)) / s
            if denom <= 0:
                denom = s
            return float(np.sqrt(max(0.0, float(np.sum(w * (v - mu) ** 2)) / denom)))

        row: dict = {}
        source = image1.metadata.source_path
        row["filename"] = str(source.name) if source else ""
        row["n_valid_slices"] = len(M1s)
        row["n_total_slices"] = n
        row["mean_M1"] = _wmean(M1s, weights)
        row["mean_M2"] = _wmean(M2s, weights)
        row["mean_pearson_r"] = _wmean(Rs, weights)
        row["mean_jaccard"] = _wmean(Js, weights)
        row["std_M1"] = _wstd(M1s, weights)
        row["std_M2"] = _wstd(M2s, weights)
        row["std_pearson_r"] = _wstd(Rs, weights)
        row["std_jaccard"] = _wstd(Js, weights)
        row["total_overlap_pixels_3d"] = total_overlap
        row["total_ch1_pixels_3d"] = total_ch1
        row["total_ch2_pixels_3d"] = total_ch2

        # Volumetric M1/M2
        if do_volumetric:
            row["volumetric_M1"] = (
                vol_ch1_coloc / vol_ch1_total if vol_ch1_total > 0 else 0.0
            )
            row["volumetric_M2"] = (
                vol_ch2_coloc / vol_ch2_total if vol_ch2_total > 0 else 0.0
            )

        # ICQ
        if do_icq:
            row["mean_ICQ"] = _wmean(icq_vals, icq_weights)
            row["std_ICQ"] = _wstd(icq_vals, icq_weights)

        # Costes summary
        if do_costes and costes_results:
            n_sig = sum(1 for c in costes_results if c["significant"])
            row["costes_n_slices_tested"] = len(costes_results)
            row["costes_n_significant"] = n_sig
            row["costes_fraction_significant"] = float(n_sig / len(costes_results))
            row["costes_mean_observed_r"] = float(
                np.mean([c["observed_r"] for c in costes_results])
            )
            row["costes_mean_scrambled_r"] = float(
                np.mean([c["mean_scrambled_r"] for c in costes_results])
            )
            row["costes_mean_p_value"] = float(
                np.mean([c["p_value"] for c in costes_results])
            )

        # Object-based 3D
        if do_object_3d and obj_3d_row:
            row.update(obj_3d_row)

        table = TableData()
        table.add_row(row)

        progress_callback(1.0)
        return {"measurements": table}

    @staticmethod
    def _to_gray_stack(data):
        if data.ndim == 2:
            return data[np.newaxis, ...]
        if data.ndim == 4:
            return data[..., 0]
        return data

    def validate_input(self, image: ImageContainer) -> tuple[bool, str]:
        if image is None or image.data is None:
            return False, "No input image provided"
        return True, ""
