"""Reuse the local TextEditBench pixel implementations without changing formulas.

The official PSNR path aligns BGR images with SIFT/FLANN/LMEDS, then computes
float32 masked MSE/PSNR. The official SSIM/LPIPS path uses unaligned RGB images,
zeroes the union of edit masks, and creates fresh TorchMetrics objects per case.
Imports and pretrained-model initialization are lazy; invalid-input CLI checks do
not need torch, LPIPS weights, or an initialized GPU.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
import math
import os
from pathlib import Path

import numpy as np

BUNDLED_EVALUATION = Path(__file__).resolve().parents[1]/'vendor/texteditbench/evaluation'
LEGACY_EVALUATION = BUNDLED_EVALUATION


def resolve_official_evaluation(official_dir=None):
    """Keep the original local source, with a portable byte-identical fallback."""
    explicit=official_dir if official_dir is not None else os.getenv('TEXTEDITBENCH_EVALUATION_DIR')
    if explicit is not None:
        folder=Path(explicit).expanduser().resolve()
        if not all((folder/name).is_file() for name in ('mse_psnr.py','lipis_ssim.py')):
            raise FileNotFoundError('Official pixel evaluation directory is missing required sources: '+str(folder))
        return folder
    for folder in (LEGACY_EVALUATION,BUNDLED_EVALUATION):
        if all((folder/name).is_file() for name in ('mse_psnr.py','lipis_ssim.py')):return folder.resolve()
    raise FileNotFoundError('Neither local nor bundled TextEditBench pixel sources are available')


OFFICIAL_EVALUATION = resolve_official_evaluation()
PROTOCOL_VERSION = 'texteditbench-official-pixels-v2'
METRICS = ('mse', 'psnr', 'ssim', 'lpips')


class OfficialMetricError(RuntimeError):
    """An official metric could not be computed; never substitute a score."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_info(official_dir: Path | str | None = None) -> dict:
    """Return provenance without importing optional metric dependencies."""
    folder = resolve_official_evaluation(official_dir)
    sources = {name: {'path': str(folder/name), 'sha256': _sha(folder/name)}
               for name in ('mse_psnr.py', 'lipis_ssim.py')}
    return {
        'version': PROTOCOL_VERSION,
        'sources': sources,
        'comparison': 'Prediction versus source image outside source/target edit-mask union.',
        'mse_psnr': 'Official SIFT/FLANN/LMEDS affine alignment; BGR; float32 masked MSE; per-case MSE/PSNR rounded to4 decimals.',
        'ssim_lpips': 'Official unaligned RGB, mask edit area to zero in both images; fresh independent metric objects per case and unchanged official calculation methods.',
        'ssim': 'Official TorchMetrics StructuralSimilarityIndexMeasure(data_range=1.0), no identity shortcut.',
        'lpips': "Official LearnedPerceptualImagePatchSimilarity(net_type='squeeze'); pretrained backbone and bundled LPIPS weights.",
        'aggregation': 'avg_mse and avg_psnr are arithmetic means of per-case4-decimal official values; avg_ssim is the arithmetic mean of unrounded per-case SSIM.',
        'avg_lpips_note': 'Arithmetic mean of official per-case LPIPS; the official lipis_ssim.py saves LPIPS per case but does not print an LPIPS aggregate.',
        'alignment_rng': 'No extra OpenCV RNG seed is set; follows official behavior.',
        'failure_policy': 'Scores fail independently: alignment affects only MSE/PSNR; SSIM and LPIPS imports, initialization and calculation errors are recorded explicitly, with no substitute scores.',
        'perceptual_initialization': 'Official metric classes and constructor parameters, initialized separately so unavailable LPIPS weights do not prevent SSIM. MetricsCalculator calculation methods are unchanged.',
        'file_policy': 'Alignment and masks apply only to arrays; raw and RGB output files are never rewritten.',
    }


@lru_cache(maxsize=8)
def _load_official(official_dir: str, stem: str):
    """Import the two independent official paths separately.

    An optional TorchMetrics/LPIPS dependency must not prevent MSE/PSNR from
    running. Failed imports are not cached and can be retried after repair.
    """
    folder = Path(official_dir)
    filename = folder/(stem+'.py')
    spec = importlib.util.spec_from_file_location('_texteditbench_official_'+stem, filename)
    if spec is None or spec.loader is None:
        raise OfficialMetricError(f'Cannot import official source {filename}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(source_rgb, prediction_rgb, source_mask, target_mask):
    source = np.asarray(source_rgb)
    prediction = np.asarray(prediction_rgb)
    if source.dtype != np.uint8 or prediction.dtype != np.uint8:
        raise ValueError('Official RGB inputs must be uint8')
    if source.ndim != 3 or source.shape[2] != 3 or prediction.shape != source.shape:
        raise ValueError('Source and prediction must have identical native RGB dimensions')
    masks = [np.asarray(source_mask), np.asarray(target_mask)]
    if any(m.dtype != np.uint8 or m.ndim != 2 or m.shape != source.shape[:2] for m in masks):
        raise ValueError('Both masks must be same-size uint8 grayscale arrays')
    return source, prediction, masks


def score_case(source_rgb, prediction_rgb, source_mask, target_mask, *,
               device='cpu', official_dir: Path | str | None = None) -> dict:
    """Compute official scores with independent, explicit per-metric failures.

    Returned keys are mse, psnr, ssim, lpips. MSE/PSNR are rounded exactly where
    the official process_dataset() rounds them before computing its means.
    SSIM/LPIPS always compare the original unaligned RGB arrays.
    A failed score is None, with its reason in metric_errors and status 'error'
    in metric_status. Common invalid input geometry/dtypes still raise.
    """
    source, prediction, masks = _inputs(source_rgb, prediction_rgb, source_mask, target_mask)
    official_dir = str(resolve_official_evaluation(official_dir))
    result = {key: None for key in METRICS}
    result['metric_status'] = {key: 'error' for key in METRICS}
    result['metric_errors'] = {}

    def failed(keys, exc):
        reason = f'{type(exc).__name__}: {exc}'
        for key in keys:
            result['metric_errors'][key] = reason

    def scored(key, value):
        try:
            if value is None:
                raise OfficialMetricError(f'Official {key.upper()} returned no score')
            value = float(value)
            if not math.isfinite(value) and not (key == 'psnr' and value == math.inf):
                raise OfficialMetricError(f'Official implementation produced non-finite {key.upper()}')
            result[key] = round(value, 4) if key in ('mse', 'psnr') else value
            result['metric_status'][key] = 'ok'
        except Exception as exc:
            failed((key,), exc)

    # OpenCV's official grayscale conversion expects BGR. A contiguous channel
    # reversal is lossless and does not modify the original arrays/files.
    import cv2
    combined_mask = cv2.bitwise_or(masks[0], masks[1])
    try:
        mse_module = _load_official(official_dir, 'mse_psnr')
        source_bgr = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
        prediction_bgr = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)
        aligned = mse_module.align_images(source_bgr, prediction_bgr)
        if aligned is None:
            raise OfficialMetricError('Official SIFT/FLANN/LMEDS image alignment failed')
        mse, psnr = mse_module.calculate_masked_metrics(source_bgr, aligned, combined_mask)
        if mse is None and psnr is None:
            raise OfficialMetricError('Official masked MSE/PSNR has no valid background pixels')
        scored('mse', mse)
        scored('psnr', psnr)
    except Exception as exc:
        failed(('mse', 'psnr'), exc)

    inverse_mask = (combined_mask < 128).astype(np.uint8)*255
    try:
        perceptual_module = _load_official(official_dir, 'lipis_ssim')
    except Exception as exc:
        failed(('ssim', 'lpips'), exc)
    else:
        for key in ('ssim', 'lpips'):
            try:
                # The official __init__ eagerly constructs LPIPS before SSIM.
                # Build its container and the requested metric separately with
                # the identical official classes/parameters, then reuse the
                # unmodified official tensor preparation and score methods.
                # New objects for every case preserve official metric state.
                calculator = perceptual_module.MetricsCalculator.__new__(perceptual_module.MetricsCalculator)
                calculator.device = device
                if key == 'ssim':
                    calculator.ssim_metric = perceptual_module.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
                else:
                    calculator.lpips_metric = perceptual_module.LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
                method = getattr(calculator, 'calculate_'+key)
                scored(key, method(prediction, source, inverse_mask, inverse_mask))
            except Exception as exc:
                failed((key,), exc)
    return result


def summarize(rows: list[dict]) -> dict:
    """Official arithmetic means; caller decides required coverage/failure policy.

    Input rows must already be valid scores. Infinities are preserved: a perfect
    PSNR remains infinity, as in the official np.mean([...]) implementation.
    """
    rows = [r for r in rows if r.get('status', 'ok') == 'ok']
    if not rows:
        return {'avg_mse': None, 'avg_psnr': None, 'avg_ssim': None, 'avg_lpips': None}
    return {'avg_'+key: float(np.mean([float(r[key]) for r in rows]))
            for key in ('mse', 'psnr', 'ssim', 'lpips')}
