"""Совместимость: синтетический генератор живёт в ml.track.synth (нужен и replay)."""
from ml.track.synth import *  # noqa: F401,F403
from ml.track.synth import (CLIENT, DOOR_PT, NEAR_SAFE, SEAT, SEAT2, Obj, Person, Scene, SynthClip,  # noqa: F401
                            affine, apply_box, apply_box_h, apply_pt, homography, make, scene_cashier_enters_client_seated,
                            scene_clean, scene_no_desk, scene_two_staff, scene_v01, scene_v02, scene_v02_outsider,
                            scene_v07, walker, without)
