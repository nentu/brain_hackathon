import math

import cv2
import numpy as np

ORB = cv2.ORB_create(
    nfeatures=3000,
    scaleFactor=1.2,
    nlevels=4,
    edgeThreshold=31,
    patchSize=31,
    fastThreshold=20,
)

BF = cv2.BFMatcher(cv2.NORM_HAMMING)


def features(gray):
    return ORB.detectAndCompute(gray, None)


def get_good_matches(kp_ref, des_ref, kp_img, des_img):
    if des_ref is None or des_img is None:
        return None, None
    if len(des_ref) < 2 or len(des_img) < 2:
        return None, None

    # Ratio test
    pairs = BF.knnMatch(des_ref, des_img, k=2)
    good = []
    for p in pairs:
        if len(p) < 2:
            continue
        m, n = p
        if m.distance < 0.75 * n.distance:
            good.append(m)

    # Fallback if ratio test gives too few matches
    if len(good) < 8:
        bf2 = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        good = bf2.match(des_ref, des_img)
        good = [m for m in good if m.distance < 64]

    if len(good) < 4:
        return None, None

    # src = candidate/flipped image, dst = reference image
    src = np.float32([kp_img[m.trainIdx].pt for m in good])
    dst = np.float32([kp_ref[m.queryIdx].pt for m in good])
    return src, dst


def try_variant(kp_ref, des_ref, gray, flip_h, flip_v):
    img = gray.copy()

    # flip_h: left-right mirror, around Y axis, cv2.flip(..., 1)
    if flip_h:
        img = cv2.flip(img, 1)

    # flip_v: up-down mirror, around X axis, cv2.flip(..., 0)
    if flip_v:
        img = cv2.flip(img, 0)

    kp, des = features(img)
    src, dst = get_good_matches(kp_ref, des_ref, kp, des)

    if src is None:
        return None

    M, inliers = cv2.estimateAffinePartial2D(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )

    if M is None:
        return None

    pred = cv2.transform(src.reshape(-1, 1, 2), M).reshape(-1, 2)
    errs = np.linalg.norm(pred - dst, axis=1)

    if inliers is not None:
        mask = inliers.ravel().astype(bool)
        n_inliers = int(mask.sum())
        if n_inliers < 6:
            return None
        reproj_error = float(np.mean(errs[mask]))
    else:
        n_inliers = len(errs)
        reproj_error = float(np.mean(errs))

    angle = math.degrees(math.atan2(M[1, 0], M[0, 0]))
    angle = (angle + 180 + 360) % 360 - 180  # normalize to [-180, 180]

    return {
        "angle_deg": angle,
        "flip_horizontal": flip_h,
        "flip_vertical": flip_v,
        "reproj_error": reproj_error,
        "inliers": n_inliers,
        "matches": len(src),
        "M": M,
    }


def adapter_image_transforms(transform: dict):
    if transform["angle_deg"] < -2:
        transform["angle_deg"] = 180 + transform["angle_deg"]
        transform["flip_horizontal"] = not transform["flip_horizontal"]
        transform["flip_vertical"] = not transform["flip_vertical"]
    return transform


def circmedian_degrees(angles):
    """
    Calculates the circular median of a list of angles in degrees.
    """
    angs = np.deg2rad(angles)

    # Create a 2D grid of pairwise differences
    pdists = angs[np.newaxis, :] - angs[:, np.newaxis]

    # Normalize differences to the [-pi, pi] range to handle wrap-around
    pdists = (pdists + np.pi) % (2 * np.pi) - np.pi

    # Find the sum of absolute distances for each angle candidate
    total_distances = np.abs(pdists).sum(axis=1)

    # Return the angle (converted back to degrees) that minimizes total distance
    median_rad = angs[np.argmin(total_distances)]
    return np.degrees(median_rad) % 360


def _flip_code(transform):
    return (1 if transform["flip_horizontal"] else 0) | (
        2 if transform["flip_vertical"] else 0
    )


def compose_transforms(first, second):
    """Return first ∘ second (apply `second` to the image first, then `first`).

    A transform is rotation applied after an optional flip, matching
    ``apply_augmentation``. Reflections conjugate the rotation, hence the sign flip.
    """
    s_first = _flip_code(first)
    s_second = _flip_code(second)
    sigma = -1 if s_first in (1, 2) else 1
    s = s_first ^ s_second
    angle = (first["angle_deg"] + sigma * second["angle_deg"] + 180) % 360 - 180
    return {
        "angle_deg": angle,
        "flip_horizontal": bool(s & 1),
        "flip_vertical": bool(s & 2),
    }


def invert_transform(transform):
    s = _flip_code(transform)
    sigma = -1 if s in (1, 2) else 1
    return {
        "angle_deg": (-sigma * transform["angle_deg"] + 180) % 360 - 180,
        "flip_horizontal": bool(s & 1),
        "flip_vertical": bool(s & 2),
    }


def _transform_distance(a, b):
    d = compose_transforms(invert_transform(a), b)
    flip_penalty = 0.0 if _flip_code(a) == _flip_code(b) else 180.0
    return abs(d["angle_deg"]) + flip_penalty


def center_transforms(transform_list):
    """Rotate the whole group so the median transform becomes the identity.

    The reference image passed to :func:`process_images` can itself be rotated or
    flipped, which would tilt every other modality. Here the medoid transform in the
    dihedral group (rotation + optional reflection) is picked and its inverse is
    left-multiplied onto every transform. The reference frame then matches the
    majority orientation instead of the (possibly augmented) reference image.
    """
    valid = [t for t in transform_list if t.get("angle_deg") is not None]
    if len(valid) < 2:
        return [dict(t) for t in transform_list]

    median = min(valid, key=lambda c: sum(_transform_distance(c, t) for t in valid))
    median_inv = invert_transform(median)

    centered_list = []
    for transform in transform_list:
        if transform.get("angle_deg") is None:
            centered_list.append(dict(transform))
            continue
        centered = compose_transforms(median_inv, transform)
        centered.update(
            {
                k: v
                for k, v in transform.items()
                if k not in ("angle_deg", "flip_horizontal", "flip_vertical")
            }
        )
        centered_list.append(centered)
    return centered_list


def process_images(img_list):
    for i in range(len(img_list)):
        if len(img_list[i].shape) != 2:
            img_list[i] = cv2.cvtColor(img_list[i], cv2.COLOR_BGR2GRAY)
    ref_gray = img_list[0]  # TODO FIX результат зависит от id.

    kp_ref, des_ref = features(ref_gray)
    results = []

    for gray in img_list:
        if gray is None:
            results.append(
                {
                    "angle_deg": None,
                    "flip_horizontal": False,
                    "flip_vertical": False,
                    "reproj_error": None,
                }
            )
            continue

        best = None

        for flip_h in (False, True):
            cand = try_variant(kp_ref, des_ref, gray, flip_h, False)
            if cand is None:
                continue

            if best is None or cand["reproj_error"] < best["reproj_error"]:
                best = cand

        if best is None:
            results.append(
                {
                    "angle_deg": None,
                    "flip_horizontal": False,
                    "flip_vertical": False,
                    "reproj_error": None,
                }
            )
        else:
            results.append(
                {
                    "angle_deg": best["angle_deg"],
                    "flip_horizontal": best["flip_horizontal"],
                    "flip_vertical": best["flip_vertical"],
                    "reproj_error": best["reproj_error"],
                    "inliers": best["inliers"],
                    "matches": best["matches"],
                }
            )

    results = [adapter_image_transforms(i) for i in results]
    results = center_transforms(results)
    return results


def apply_augmentation(image: np.ndarray, aug: dict) -> np.ndarray:
    angle = float(aug.get("angle_deg", 0.0))
    flip_h = bool(aug.get("flip_horizontal", False))
    flip_v = bool(aug.get("flip_vertical", False))

    h, w = image.shape[:2]

    if flip_h and flip_v:
        image = cv2.flip(image, -1)  # обе оси
    elif flip_h:
        image = cv2.flip(image, 1)  # горизонтально
    elif flip_v:
        image = cv2.flip(image, 0)  # вертикально
    if angle != 0.0:
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, -angle, scale=1.0)
        image = cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    return image
