"""Kernel SHAP heatmaps for the best models."""

import math
import os

import cv2
import numpy as np
import torch

from brain_hackathon.eval.base import DEFAULT_CLASSES
from brain_hackathon.eval.gradcam import _display_base, _resolve_target
from brain_hackathon.eval.masking import apply_heatmap_mask

__all__ = ["shap", "sharp"]


def _patch_bounds(size, count):
    edges = np.linspace(0, size, count + 1).astype(int)
    return [(int(edges[i]), int(edges[i + 1])) for i in range(count)]


def _prepare_baseline(baseline, tensor):
    base = torch.as_tensor(baseline, dtype=tensor.dtype, device=tensor.device)
    if base.ndim == 0:
        return torch.full_like(tensor, float(base))
    if base.ndim == 2:
        base = base.unsqueeze(0)
    try:
        base = torch.broadcast_to(base, tensor.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"baseline shape {tuple(base.shape)} is not broadcastable "
            f"to model input {tuple(tensor.shape)}"
        ) from exc
    return base.clone()


def _mask_tensor(tensor, keep, row_bounds, col_bounds, baseline):
    masked = tensor.clone()
    columns = len(col_bounds)
    for position, present in enumerate(keep):
        if present:
            continue
        row, column = divmod(position, columns)
        row_start, row_end = row_bounds[row]
        column_start, column_end = col_bounds[column]
        masked[:, row_start:row_end, column_start:column_end] = baseline[
            :, row_start:row_end, column_start:column_end
        ]
    return masked


def _predict_values(
    network,
    tensor,
    masks,
    target_index,
    row_bounds,
    col_bounds,
    baseline,
    batch_size,
):
    values = []
    for start in range(0, len(masks), batch_size):
        batch = torch.stack(
            [
                _mask_tensor(tensor, mask, row_bounds, col_bounds, baseline)
                for mask in masks[start : start + batch_size]
            ]
        )
        with torch.no_grad():
            probabilities = torch.softmax(network(batch), dim=1)
        values.append(probabilities[:, target_index].cpu().numpy())
    return np.concatenate(values)


def _coalition_masks(n_samples, n_patches, seed):
    rng = np.random.default_rng(seed)
    masks = rng.integers(0, 2, size=(n_samples, n_patches), dtype=np.int8).astype(bool)
    masks[0] = False
    masks[1] = True
    masks = np.unique(masks, axis=0)
    if len(masks) < n_patches + 1:
        raise ValueError(
            f"n_samples={n_samples} produced only {len(masks)} unique "
            f"coalitions; need at least n_patches + 1 = {n_patches + 1}"
        )
    return masks


def _kernel_weights(masks, n_features):
    sizes = masks.sum(axis=1)
    weights = np.zeros(len(masks), dtype=np.float64)
    for size in range(1, n_features):
        selected = sizes == size
        if selected.any():
            weights[selected] = (n_features - 1) / (
                size * (n_features - size) * math.comb(n_features, size)
            )
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError("could not compute Kernel SHAP weights")
    return weights / total


def _solve_shapley(masks, values, n_features):
    """Solve the Kernel SHAP weighted least-squares problem."""
    masks = np.asarray(masks, dtype=bool)
    values = np.asarray(values, dtype=np.float64)
    if masks.ndim != 2 or masks.shape[1] != n_features:
        raise ValueError(
            f"expected masks of shape (N, {n_features}), got {masks.shape}"
        )
    if len(values) != len(masks):
        raise ValueError("masks and values must have the same length")

    empty_rows = np.where(~masks.any(axis=1))[0]
    full_rows = np.where(masks.all(axis=1))[0]
    if len(empty_rows) == 0 or len(full_rows) == 0:
        raise ValueError("masks must include the empty and full coalitions")

    value_empty = float(values[empty_rows[0]])
    value_full = float(values[full_rows[0]])
    if n_features == 1:
        return np.array([value_full - value_empty]), value_empty, value_full

    weights = _kernel_weights(masks, n_features)
    design = masks.astype(np.float64)
    target = values - value_empty

    normal = (design * weights[:, None]).T @ design
    gradient = (design * weights[:, None]).T @ target
    normal += np.eye(n_features) * 1e-8

    system = np.zeros((n_features + 1, n_features + 1), dtype=np.float64)
    system[:n_features, :n_features] = normal
    system[:n_features, n_features] = 1.0
    system[n_features, :n_features] = 1.0
    rhs = np.concatenate([gradient, [value_full - value_empty]])

    phi = np.linalg.lstsq(system, rhs, rcond=None)[0][:n_features]
    return phi, value_empty, value_full


def shap(
    model,
    image,
    target_class=None,
    *,
    base_image=None,
    base_modality=None,
    patch_grid=7,
    n_samples=256,
    batch_size=64,
    baseline=0.0,
    seed=0,
    alpha=0.45,
    colormap=cv2.COLORMAP_JET,
    output_path=None,
    mask=False,
    mask_radius=10.0,
):
    """Draw a Kernel SHAP heatmap for one image.

    This is a sampling-based Kernel SHAP approximation over a grid of image
    patches. Each patch is a player; absent patches are replaced by ``baseline``
    in the preprocessed model input. The heatmap contains normalized absolute
    Shapley values, so both supporting and opposing regions are visible.

    ``model`` is a :class:`BaseBestModel` wrapper (raw or aligned). ``image`` is a
    single raw image, an aligned ``HxWx6`` array, or a modality sequence/mapping.
    ``target_class`` is ``None`` (predicted class), a class name, or an index.
    ``patch_grid`` is an int or ``(rows, cols)``; ``n_samples`` is the number of
    sampled coalitions and must be at least ``rows * cols + 1``.

    The background is the same as for Grad-CAM++: the input image for raw models
    and the mean of non-empty aligned modalities by default. Pass ``base_modality``
    or ``base_image`` to override it. If ``mask`` is true, the heatmap is
    restricted to the largest non-background region before it is drawn;
    ``mask_radius`` controls the intensity tolerance in the ``[0, 255]`` scale.

    Returns a dict with ``heatmap`` (float32 HxW in ``[0, 1]``), ``overlay``
    (uint8 HxWx3 BGR), ``class_index``, ``class_name``, and ``probability``.
    If ``output_path`` is given, the overlay is written there with ``cv2.imwrite``.
    """
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    if isinstance(patch_grid, (int, np.integer)):
        rows = columns = int(patch_grid)
    else:
        try:
            rows, columns = patch_grid
        except (TypeError, ValueError) as exc:
            raise ValueError("patch_grid must be an int or (rows, cols)") from exc
    rows, columns = int(rows), int(columns)
    if rows < 1 or columns < 1:
        raise ValueError("patch_grid dimensions must be positive")

    n_samples = int(n_samples)
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    network = getattr(model, "model", None)
    if network is None:
        raise TypeError("model must expose a 'model' network attribute")

    parameters = list(network.parameters())
    device = parameters[0].device if parameters else getattr(model, "device", "cpu")

    tensor = model.preprocess(image)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.to(device).float()
    if tensor.ndim != 3:
        raise ValueError(
            f"preprocess must return a (C, H, W) tensor, got {tuple(tensor.shape)}"
        )

    channels, height, width = tensor.shape
    if rows > height or columns > width:
        raise ValueError(
            f"patch_grid={(rows, columns)} exceeds input size {(height, width)}"
        )

    n_patches = rows * columns
    if n_samples < n_patches + 1:
        raise ValueError(
            f"n_samples={n_samples} is too small for {n_patches} patches; "
            f"need at least {n_patches + 1}"
        )

    row_bounds = _patch_bounds(height, rows)
    column_bounds = _patch_bounds(width, columns)
    baseline_tensor = _prepare_baseline(baseline, tensor)
    classes = tuple(getattr(model, "classes", DEFAULT_CLASSES))

    was_training = network.training
    try:
        network.eval()
        with torch.no_grad():
            logits = network(tensor.unsqueeze(0))
            target_index = _resolve_target(classes, target_class, logits)
            probability = float(torch.softmax(logits, dim=1)[0, target_index].item())

        masks = _coalition_masks(n_samples, n_patches, seed)
        values = _predict_values(
            network,
            tensor,
            masks,
            target_index,
            row_bounds,
            column_bounds,
            baseline_tensor,
            batch_size,
        )
    finally:
        network.train(was_training)

    phi, _, _ = _solve_shapley(masks, values, n_patches)
    if not np.isfinite(phi).all():
        raise RuntimeError("SHAP solver produced non-finite values")

    grid = np.abs(phi).reshape(rows, columns).astype(np.float32)
    high = float(grid.max())
    if high > 0.0:
        grid = grid / high
    else:
        grid = np.zeros_like(grid)

    heatmap = cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)
    heatmap = np.clip(heatmap, 0.0, 1.0).astype(np.float32)

    base = _display_base(image, int(channels), base_image, base_modality)
    base_height, base_width = base.shape[:2]
    if heatmap.shape != (base_height, base_width):
        heatmap = cv2.resize(
            heatmap, (base_width, base_height), interpolation=cv2.INTER_LINEAR
        )
    if mask:
        heatmap = apply_heatmap_mask(heatmap, base, mask_radius)

    colored = cv2.applyColorMap(np.uint8(np.rint(heatmap * 255.0)), colormap)
    blended = cv2.addWeighted(colored, alpha, base, 1.0 - alpha, 0)
    if mask:
        overlay = base.copy()
        active = heatmap > 0.0
        overlay[active] = blended[active]
    else:
        overlay = blended

    if output_path is not None:
        path = os.fspath(output_path)
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not cv2.imwrite(path, overlay):
            raise OSError(f"failed to write heatmap to {path!r}")

    return {
        "heatmap": heatmap,
        "overlay": overlay,
        "class_index": int(target_index),
        "class_name": classes[target_index],
        "probability": probability,
    }


sharp = shap
