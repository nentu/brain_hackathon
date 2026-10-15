import albumentations as A
import cv2

train_aug = A.Compose(
    [
        # Геометрические
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
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
        A.KSpaceSpikeNoise(
            spike_amplitude=(0.01, 0.05), spike_frequency=(0.01, 0.05), p=0.05
        ),
    ]
)

val_aug = A.Compose(
    [
        A.Resize(224, 224),
    ]
)
