import numpy as np

from ml.vlm.frames import (annotate, get_frame, make_grid, motion_energy, prepare_images, resize_max_side,
                           to_pil_rgb, verify_times)


def test_verify_times_include_peak_and_clamp():
    ts = verify_times(10.0, 12.0, 11.3, duration=60.0, n=5, pad=1.0)
    assert len(ts) == 5 and 11.3 in ts and ts[0] == 9.0 and ts[-1] == 13.0
    ts = verify_times(0.0, 0.5, 0.2, duration=60.0, n=5, pad=1.0)
    assert ts[0] == 0.0 and all(t >= 0 for t in ts)
    ts = verify_times(58.0, 60.0, 59.5, duration=60.0, n=4, pad=1.0)
    assert max(ts) <= 60.0
    ts = verify_times(0.0, 100.0, 50.0, duration=145.0, n=6, pad=1.0, max_span=12.0)
    assert ts[-1] - ts[0] <= 12.0 + 1e-6 and 50.0 in ts


def test_to_pil_bgr_to_rgb():
    f = np.zeros((10, 20, 3), np.uint8)
    f[:, :, 0] = 255  # синий в BGR
    img = to_pil_rgb(f)
    assert img.size == (20, 10) and img.getpixel((0, 0)) == (0, 0, 255)


def test_resize_and_annotate():
    img = to_pil_rgb(np.zeros((848, 464, 3), np.uint8))
    r = resize_max_side(img, 448)
    assert max(r.size) == 448 and r.size[0] == 245
    a = annotate(r, {"cell_phone": (0.1, 0.1, 0.5, 0.5), "bad": (2, 2, 3, 3), "junk": "x"}, ts=1.5, index=2)
    arr = np.asarray(a)
    assert (arr[:, :, 0] > 200).any()  # красная рамка нарисована
    assert np.asarray(r).sum() == 0  # исходник не испорчен


def test_prepare_images_multi_and_grid():
    frames = [np.zeros((848, 464, 3), np.uint8) for _ in range(5)]
    times = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = prepare_images(frames, times, {"person": (0.2, 0.2, 0.6, 0.9)}, 3.0, {"max_side": 448})
    assert len(out) == 5 and all(max(i.size) <= 448 for i in out)
    grid = prepare_images(frames, times, None, None, {"max_side": 448, "image_mode": "grid", "grid_cols": 3})
    assert len(grid) == 1 and max(grid[0].size) <= 896
    assert make_grid(out, 3).size[0] > out[0].size[0]


def test_get_frame_fallback_to_sampled(clip):
    class Broken:
        times = [0.0, 0.2, 0.4]
        def frame_at(self, t):
            raise RuntimeError("no exact seek")
        def frame(self, i):
            return np.full((4, 4, 3), i, np.uint8)
    f = get_frame(Broken(), 0.35)
    assert int(f[0, 0, 0]) == 2
    assert get_frame(clip, 3.0).shape == (848, 464, 3)


def test_motion_energy_detects_activity():
    from ml.vlm.tests.conftest import SyntheticClip
    c = SyntheticClip(duration=10.0, moving_from=4.0, moving_to=7.0)
    me = motion_energy(c, downscale=64, target_fps=2.0)
    assert me and all(0.0 <= e <= 1.0 for _, e in me)
    active = [e for t, e in me if 4.5 <= t <= 7.0]
    idle = [e for t, e in me if t <= 3.5 or t >= 8.0]
    assert max(active) > max(idle)
