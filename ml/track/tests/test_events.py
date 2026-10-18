from ml.track.contract import Track, Zone
from ml.track.events import dwell, entries_exits, gaps, person_signals, presence_state, proximity

Z = Zone("cashier_area", [(0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)], 0.9)
DOOR = Zone("door", [(0.0, 0.0), (0.15, 0.0), (0.15, 0.2), (0.0, 0.2)], 0.8)


def _box(x, y, s=0.1):
    return (x - s / 2, y - s / 2, x + s / 2, y + s / 2)


def _walk(points, dt=0.2, tid=1, role="cashier", t0=0.0):
    return Track(tid, "person", role, 0.9, [(round(t0 + i * dt, 4), _box(x, y)) for i, (x, y) in enumerate(points)])


def test_entries_exits_with_hysteresis():
    pts = [(0.2, 0.5)] * 5 + [(0.5, 0.5)] * 10 + [(0.8, 0.5)] * 5
    ev = entries_exits([_walk(pts)], [Z])
    assert [(e[2], e[3]) for e in ev] == [("cashier_area", "enter"), ("cashier_area", "exit")]
    assert abs(ev[0][0] - 1.0) < 1e-6 and abs(ev[1][0] - 3.0) < 1e-6
    flick = [(0.5, 0.5)] * 5 + [(0.8, 0.5)] + [(0.5, 0.5)] * 5  # 1 кадр снаружи — не выход
    ev2 = entries_exits([_walk(flick)], [Z])
    assert [e[3] for e in ev2] == ["enter"]


def test_gap_is_not_exit_by_default():
    tr = Track(1, "person", "cashier", 0.9, [(i * 0.2, _box(0.5, 0.5)) for i in range(5)] +
               [(10 + i * 0.2, _box(0.5, 0.5)) for i in range(5)])
    assert [e[3] for e in entries_exits([tr], [Z])] == ["enter"]
    assert [e[3] for e in entries_exits([tr], [Z], gap_as_exit_s=3.0)] == ["enter", "exit", "enter"]


def test_gaps_classification():
    inner = Track(1, "person", "cashier", 0.9, [(0.0, _box(0.5, 0.5)), (0.2, _box(0.5, 0.5)), (5.0, _box(0.5, 0.5))])
    g = gaps(inner, [Z, DOOR])
    assert g[0]["kind"] == "lost_in_view" and g[0]["last_zone"] == "cashier_area"
    edge = Track(2, "person", "visitor", 0.9, [(0.0, _box(0.5, 0.5)), (0.2, _box(0.97, 0.5))])
    assert gaps(edge, [Z], t_end=10.0)[-1]["kind"] == "left_view_edge"
    door = Track(3, "person", "visitor", 0.9, [(0.0, _box(0.5, 0.5)), (0.2, _box(0.07, 0.1))])
    assert gaps(door, [Z, DOOR], t_end=10.0)[-1]["kind"] == "left_via_door"
    assert presence_state(inner, 2.0, [Z]) == "lost_in_view"
    assert presence_state(inner, 0.1, [Z]) == "observed"


def test_dwell_and_proximity():
    tr = _walk([(0.5, 0.5)] * 10)
    assert abs(dwell(tr, Z) - 2.0) < 1e-6
    other = _walk([(0.55, 0.5)] * 10, tid=2, role="visitor")
    far = _walk([(0.95, 0.95)] * 10, tid=3, role="visitor")
    assert proximity(tr, other, 1.0) == 1.0
    assert proximity(tr, far, 1.0) < 0.2
    assert proximity(tr, other, 50.0) == 0.0


def test_person_signals_occluded_cashier_is_uncertain():
    cashier = Track(1, "person", "cashier", 0.9, [(0.0, _box(0.5, 0.5)), (0.2, _box(0.5, 0.5)), (5.0, _box(0.5, 0.5))])
    vis = Track(2, "person", "visitor", 0.8, [(2.0, _box(0.5, 0.45))])
    st = person_signals([cashier, vis], [Z], [0.0, 2.0], extended=True)
    assert st[0].signals["cashier_present"] == 1.0 and st[0].signals["cashier_in_workplace"] == 1.0
    assert st[1].signals["cashier_present"] == 0.5 and st[1].signals["x_cashier_lost_in_view"] == 1.0
    assert st[1].signals["outsider_in_cashier_area"] == 0.8 and st[1].signals["n_persons"] == 1.0
    collector = Track(3, "person", "collector", 0.7, [(2.0, _box(0.5, 0.45))])
    st2 = person_signals([cashier, collector], [Z], [2.0])
    assert st2[0].signals["outsider_in_cashier_area"] == 0.0 and st2[0].signals["collector_present"] == 1.0
