import numpy as np
import pytest

from ml.track.geometry import (clip_polygon_unit, convex_hull, point_in_polygon, polygon_area,
                               polygon_centroid, reflect_polygon)
from ml.track.tests import synth as S
from ml.track.tests._util import role_of, run


def _c(z):
    return np.array(polygon_centroid(z.polygon))


def test_zones_present_and_sane():
    res, ids, by_id, zones, _ = run(S.scene_clean())
    for name in ("desk", "cashier_area", "client_area", "safe", "door"):
        assert name in zones, name
        assert 0.0 < polygon_area(zones[name].polygon) < 0.5  # никогда не «весь кадр»
        assert 0.0 <= zones[name].conf <= 1.0
    assert point_in_polygon(S.SEAT, zones["cashier_area"].polygon)
    assert point_in_polygon(S.CLIENT, zones["client_area"].polygon)
    assert point_in_polygon((0.84, 0.84), zones["safe"].polygon)
    assert point_in_polygon(S.DOOR_PT, zones["door"].polygon)
    d, ca, cl = _c(zones["desk"]), _c(zones["cashier_area"]), _c(zones["client_area"])
    assert np.dot(ca - d, cl - d) < 0  # клиентская сторона — противоположная


def test_zone_provenance_in_info():
    res, *_ = run(S.scene_clean())
    zi = res.info["zones"]
    assert zi["desk"]["method"] == "objects" and zi["desk"]["independent"] is True
    assert zi["cashier_area"]["independent"] is False and zi["cashier_area"]["cashier_support"] == "desk_objects"


@pytest.mark.perturbation
@pytest.mark.parametrize("theta,scale,flip", [(25.0, 0.8, False), (-35.0, 0.65, False), (15.0, 0.85, True)])
def test_zones_follow_camera_affine(theta, scale, flip):
    A = S.affine(theta, scale, flip_x=flip)
    _, _, _, z0, _ = run(S.scene_v01())
    res, ids, by_id, z1, _ = run(S.scene_v01(), A=A)
    assert role_of(ids, by_id, "cashier")[0] == ["cashier"]
    assert role_of(ids, by_id, "outsider")[0][0] != "cashier"
    for name in ("desk", "safe", "cashier_area"):
        exp = np.array(S.apply_pt(A, *_c(z0[name])))
        assert np.linalg.norm(_c(z1[name]) - exp) < 0.06, (name, _c(z1[name]), exp)
    assert point_in_polygon(S.apply_pt(A, *S.SEAT), z1["cashier_area"].polygon)
    d, ca, cl = _c(z1["desk"]), _c(z1["cashier_area"]), _c(z1["client_area"])
    assert np.dot(ca - d, cl - d) < 0


@pytest.mark.perturbation
def test_rotation_90_to_landscape():
    A = np.array([[0.0, -1.0, 1.0], [1.0, 0.0, 0.0]])  # x' = 1 - y, y' = x
    res, ids, by_id, z, _ = run(S.scene_clean(), A=A, width=848, height=464)
    assert role_of(ids, by_id, "cashier")[0] == ["cashier"]
    assert role_of(ids, by_id, "visitor")[0] == ["visitor"]
    assert point_in_polygon(S.apply_pt(A, *S.SEAT), z["cashier_area"].polygon)
    assert point_in_polygon(S.apply_pt(A, *S.CLIENT), z["client_area"].polygon)


def test_geometry_helpers():
    sq = convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)])
    assert len(sq) == 4 and abs(polygon_area(sq) - 1.0) < 1e-9
    cl = clip_polygon_unit([(-0.5, -0.5), (0.5, -0.5), (0.5, 0.5), (-0.5, 0.5)])
    assert abs(polygon_area(cl) - 0.25) < 1e-9
    r = reflect_polygon([(0.6, 0.6)], (0.5, 0.5))
    assert np.allclose(r, [(0.4, 0.4)])


@pytest.mark.perturbation
@pytest.mark.parametrize("k", [0.25, -0.3])
def test_perspective_tilt_keeps_roles_and_zones(k):
    H = S.homography(k)
    res, ids, by_id, z, _ = run(S.scene_v07(), H=H)
    assert role_of(ids, by_id, "cashier")[0] == ["cashier"]
    assert role_of(ids, by_id, "collector")[0] == ["collector"]
    seat = S.apply_box_h(H, (S.SEAT[0] - 0.01, S.SEAT[1] - 0.01, S.SEAT[0] + 0.01, S.SEAT[1] + 0.01))
    assert point_in_polygon(((seat[0] + seat[2]) / 2, (seat[1] + seat[3]) / 2), z["cashier_area"].polygon)


@pytest.mark.perturbation
def test_mirror_keeps_client_visitor_and_cashier():
    A = S.affine(0.0, 1.0, flip_x=True)
    res, ids, by_id, z, _ = run(S.scene_cashier_enters_client_seated(), A=A)
    assert role_of(ids, by_id, "cashier")[0] == ["cashier"]
    assert role_of(ids, by_id, "client")[0] == ["visitor"]
