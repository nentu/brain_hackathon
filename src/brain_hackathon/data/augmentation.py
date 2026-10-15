import albumentations as A
import cv2
import numpy as np


class ChannelDropout(A.ImageOnlyTransform):
    """Set whole channels to black to simulate missing modalities.

    Each channel is dropped independently with probability ``drop_prob``. With
    ``fill_value=0`` a dropped channel becomes black.
    """

    def __init__(self, drop_prob=0.3, fill_value=0, p=1.0):
        super().__init__(p=p)
        self.drop_prob = drop_prob
        self.fill_value = fill_value

    def get_params_dependent_on_data(self, params, data):
        channels = data["image"].shape[2] if data["image"].ndim == 3 else 1
        return {"drop_mask": np.random.random(channels) < self.drop_prob}

    def apply(self, img, drop_mask=None, **params):
        if drop_mask is None or not drop_mask.any():
            return img
        img = img.copy()
        if img.ndim == 3:
            for channel in range(img.shape[2]):
                if drop_mask[channel]:
                    img[:, :, channel] = self.fill_value
        else:
            img[:] = self.fill_value
        return img

    def get_transform_init_args_names(self):
        return ("drop_prob", "fill_value")


train_aug = A.Compose(
    [
        # Геометрические
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,
            p=0.7,
        ),
        A.HorizontalFlip(p=0.5),
        A.RandomResizedCrop(
            size=(224, 224), scale=(0.9, 1.0), ratio=(0.95, 1.05), p=0.5
        ),
        # Интенсивностные
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.RandomGamma(gamma_limit=(90, 110), p=0.3),
        # Шум и размытие — минимальные
        A.GaussNoise(std_range=(0.001, 0.01), p=0.2),
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 0.3), p=0.1),
        # Артефакты МРТ — минимальные
        A.RingingOvershoot(blur_limit=(7, 9), cutoff=(1.2, 1.4), p=0.05),
        # Пропущенные модальности
        ChannelDropout(drop_prob=0.2, p=1.0),
    ]
)

val_aug = A.Compose(
    [
        A.Resize(224, 224),
    ]
)
