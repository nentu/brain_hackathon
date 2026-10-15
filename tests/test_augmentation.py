import numpy as np
import pytest

from brain_hackathon.data.augmentation import ChannelDropout, train_aug

CHANNELS = 6


def _image(value=255, shape=(8, 8)):
    return np.full((*shape, CHANNELS), value, dtype=np.uint8)


def test_channel_dropout_rate():
    np.random.seed(0)
    transform = ChannelDropout(drop_prob=0.3, p=1.0)
    image = _image()
    trials = 4000

    dropped = 0
    for _ in range(trials):
        out = transform(image=image)["image"]
        dropped += sum(int(np.all(out[:, :, c] == 0)) for c in range(CHANNELS))

    rate = dropped / (trials * CHANNELS)
    assert rate == pytest.approx(0.3, abs=0.02)


def test_channel_dropout_prob_one_blacks_everything():
    out = ChannelDropout(drop_prob=1.0, p=1.0)(image=_image())["image"]
    assert np.all(out == 0)


def test_channel_dropout_prob_zero_is_noop():
    image = _image()
    out = ChannelDropout(drop_prob=0.0, p=1.0)(image=image)["image"]
    assert np.array_equal(out, image)


def test_channel_dropout_blacks_whole_channel_only():
    np.random.seed(1)
    image = _image()
    transform = ChannelDropout(drop_prob=0.5, p=1.0)
    for _ in range(20):
        out = transform(image=image)["image"]
        for c in range(CHANNELS):
            channel = out[:, :, c]
            assert np.all(channel == 0) or np.all(channel == 255)


def test_train_aug_preserves_channels():
    image = _image(shape=(64, 64))
    out = train_aug(image=image)["image"]
    assert out.ndim == 3 and out.shape[2] == CHANNELS
    assert out.dtype == np.uint8
