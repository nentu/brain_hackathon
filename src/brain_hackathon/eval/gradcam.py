"""Grad-CAM++ heatmaps for the best models."""

import os

import cv2
import numpy as np
import torch

from brain_hackathon.data.dataset import MODALITIES
from brain_hackathon.data.preprocessing import IMAGE_SIZE, resize_image
from brain_hackathon.eval.aligned_model import align_modalities
from brain_hackathon.eval.base import ALIGNED_CHANNELS, DEFAULT_CLASSES, to_gray

__all__ = ["grad_cam_plus_plus"]


def _resolve_target(classes, target_class, logits):
    if target_class is None:
        return int(logits.argmax(dim=1).item())
    if isinstance(target_class, str):
        try:
            return classes.index(target_class)
        except ValueError as exc:
            raise ValueError(
                f"unknown class {target_class!r}; expected one of {classes}"
            ) from exc
    if isinstance(target_class, (int, np.integer)):
        index = int(target_class)
        if 0 <= index < len(classes):
            return index
        raise ValueError(
            f"class index {index} is out of range for {len(classes)} classes"
        )
    raise TypeError("target_class must be None, a class name, or an index")


def _resolve_layer(model, layer):
    if layer is None:
        backbone = getattr(model, "backbone", None)
        if backbone is None or len(backbone) == 0:
            raise ValueError(
                "model has no backbone; pass layer=<module or dotted name>"
            )
        return backbone[-1]
    if isinstance(layer, torch.nn.Module):
        return layer
    if isinstance(layer, str):
        try:
            return model.get_submodule(layer)
        except AttributeError as exc:
            raise ValueError(f"unknown layer {layer!r}") from exc
    raise TypeError("layer must be a module, a dotted name, or None")


def _aligned_stack(image):
    if isinstance(image, (list, tuple, dict)):
        return align_modalities(image)

    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == ALIGNED_CHANNELS:
        return np.stack(
            [
                resize_image(array[:, :, channel], IMAGE_SIZE)
                for channel in range(ALIGNED_CHANNELS)
            ],
            axis=-1,
        )

    gray = resize_image(to_gray(array), IMAGE_SIZE)
    return np.repeat(gray[:, :, None], ALIGNED_CHANNELS, axis=2)


def _aligned_base(stack, base_modality):
    stack = np.asarray(stack)
    if stack.ndim != 3 or stack.shape[2] != ALIGNED_CHANNELS:
        raise ValueError(
            f"expected an HxWx{ALIGNED_CHANNELS} aligned stack, got shape {stack.shape}"
        )

    if base_modality is not None:
        name = str(base_modality).upper()
        if name not in MODALITIES:
            raise ValueError(
                f"unknown modality {base_modality!r}; expected one of {MODALITIES}"
            )
        channel = stack[:, :, MODALITIES.index(name)]
        if not np.any(channel != 0):
            raise ValueError(f"modality {name} is missing or empty")
        return channel

    present = [
        index for index in range(ALIGNED_CHANNELS) if np.any(stack[:, :, index] != 0)
    ]
    if not present:
        raise ValueError("cannot build display base: all aligned channels are empty")
    return stack[:, :, present].mean(axis=2)


def _to_uint8(array):
    array = np.asarray(array)
    if array.dtype == np.uint8:
        return array

    values = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if values.size and float(values.max()) <= 1.0 and float(values.min()) >= 0.0:
        values = values * 255.0
    return np.clip(values, 0.0, 255.0).astype(np.uint8)


def _display_base(image, channels, base_image, base_modality):
    if base_image is not None:
        gray = to_gray(base_image)
    elif channels == ALIGNED_CHANNELS:
        gray = _aligned_base(_aligned_stack(image), base_modality)
    else:
        if base_modality is not None:
            raise ValueError(
                "base_modality is only supported for aligned 6-channel models"
            )
        gray = to_gray(image)

    gray = resize_image(gray, IMAGE_SIZE)
    return cv2.cvtColor(_to_uint8(gray), cv2.COLOR_GRAY2BGR)


def _grad_cam_plus_plus(activations, gradients):
    activations = activations.float()
    gradients = gradients.float()

    gradients_power_2 = gradients.pow(2)
    gradients_power_3 = gradients_power_2 * gradients
    sum_activations = activations.sum(dim=(1, 2), keepdim=True)

    alpha = gradients_power_2 / (
        2.0 * gradients_power_2 + sum_activations * gradients_power_3 + 1e-7
    )
    alpha = torch.where(gradients != 0, alpha, torch.zeros_like(alpha))
    weights = (torch.relu(gradients) * alpha).sum(dim=(1, 2))

    cam = torch.relu((weights[:, None, None] * activations).sum(dim=0))
    return cam.detach().cpu().numpy()


def grad_cam_plus_plus(
    model,
    image,
    target_class=None,
    *,
    base_image=None,
    base_modality=None,
    layer=None,
    alpha=0.45,
    colormap=cv2.COLORMAP_JET,
    output_path=None,
):
    """Draw a Grad-CAM++ heatmap for one image.

    ``model`` is a :class:`BaseBestModel` wrapper (raw or aligned). ``image`` is a
    single raw image, an aligned ``HxWx6`` array, or a modality sequence/mapping.
    ``target_class`` is ``None`` (predicted class), a class name, or an index.

    The heatmap is blended over a grayscale background: the input image for raw
    models, and by default the mean of the non-empty aligned modalities. Pass
    ``base_modality`` (e.g. ``"BR"``) to overlay a specific aligned channel, or
    ``base_image`` to supply an explicit grayscale background.

    Returns a dict with ``heatmap`` (float32 HxW in ``[0, 1]``), ``overlay``
    (uint8 HxWx3 BGR), ``class_index``, ``class_name``, and ``probability``.
    If ``output_path`` is given, the overlay is written there with ``cv2.imwrite``.
    """
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    network = getattr(model, "model", None)
    if network is None:
        raise TypeError("model must expose a 'model' network attribute")

    parameters = list(network.parameters())
    device = parameters[0].device if parameters else getattr(model, "device", "cpu")

    tensor = model.preprocess(image)
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    tensor = tensor.to(device).unsqueeze(0).float().requires_grad_(True)

    classes = tuple(getattr(model, "classes", DEFAULT_CLASSES))
    target_layer = _resolve_layer(network, layer)

    activations = {}

    def capture(module, inputs, output):
        activations["value"] = output

    handle = target_layer.register_forward_hook(capture)
    was_training = network.training
    try:
        network.eval()
        with torch.enable_grad():
            logits = network(tensor)
            if "value" not in activations:
                raise RuntimeError("target layer did not run during the forward pass")
            target_index = _resolve_target(classes, target_class, logits)
            probability = float(torch.softmax(logits, dim=1)[0, target_index].item())
            gradients = torch.autograd.grad(
                logits[0, target_index], activations["value"]
            )[0]
    finally:
        handle.remove()
        network.train(was_training)

    cam = _grad_cam_plus_plus(activations["value"].detach()[0], gradients.detach()[0])
    base = _display_base(image, int(tensor.shape[1]), base_image, base_modality)

    height, width = base.shape[:2]
    heatmap = cv2.resize(cam, (width, height), interpolation=cv2.INTER_LINEAR)
    low = float(heatmap.min())
    high = float(heatmap.max())
    if high > low:
        heatmap = (heatmap - low) / (high - low)
    else:
        heatmap = np.zeros_like(heatmap)
    heatmap = np.clip(heatmap, 0.0, 1.0).astype(np.float32)

    colored = cv2.applyColorMap(np.uint8(np.rint(heatmap * 255.0)), colormap)
    overlay = cv2.addWeighted(colored, alpha, base, 1.0 - alpha, 0)

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
