"""API-ключ, UI-сессия (HttpOnly cookie + CSRF), rate-limit, отсутствие секретов в логах."""
from __future__ import annotations

import logging

from backend.api.security import ApiKeyAuth, RateLimiter, UiTokenSigner, client_ip, cross_site

from .conftest import FTYP_ONLY, H, KEY, ok_run, spec, upload, wait_job


def test_api_401_without_and_with_wrong_key(client_factory):
    with client_factory(spec(ok_run)) as c:
        for headers in ({}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}, {"Authorization": "Bearer nope"}):
            for method, path in (("post", "/api/v1/analyze"), ("get", "/api/v1/jobs/abcdef123456"),
                                 ("get", "/api/v1/jobs/abcdef123456/video"),
                                 ("get", "/api/v1/jobs/abcdef123456/frames/0.jpg"),
                                 ("post", "/api/v1/jobs/abcdef123456/ask"), ("get", "/api/v1/jobs")):
                r = getattr(c, method)(path, headers=headers)
                assert r.status_code == 401, (method, path, headers, r.text)
                assert r.headers["content-type"].startswith("application/json")
                assert "detail" in r.json()
        assert c.get("/api/v1/jobs", headers={"Authorization": f"Bearer {KEY}"}).status_code == 200


def test_ui_routes_need_session_not_origin(client_factory):
    """same-origin заголовки — НЕ аутентификация: без cookie всё закрыто."""
    with client_factory(spec(ok_run)) as c:
        same = {"Sec-Fetch-Site": "same-origin", "Origin": "http://testserver"}
        assert c.get("/ui/v1/session", headers=same).status_code == 401
        for method, path in (("post", "/ui/v1/analyze"), ("get", "/ui/v1/jobs"),
                             ("get", "/ui/v1/jobs/abcdef123456/video"),
                             ("get", "/ui/v1/jobs/abcdef123456/frames/0.jpg"),
                             ("post", "/ui/v1/jobs/abcdef123456/ask")):
            assert getattr(c, method)(path, headers=same).status_code == 401, path
        # API-ключ в заголовке не открывает /ui/v1 (там только cookie)
        assert c.get("/ui/v1/jobs", headers=H).status_code == 401


def _login(c, **body):
    return c.post("/ui/v1/session", json=body, headers={"Sec-Fetch-Site": "same-origin"})


def test_ui_login_with_key_sets_httponly_cookie(client_factory):
    with client_factory(spec(ok_run)) as c:
        assert _login(c, api_key="wrong").status_code == 401
        assert _login(c, password="").status_code == 401  # пароль не задан -> вход по паролю выключен
        r = _login(c, api_key=KEY)
        assert r.status_code == 200
        body = r.json()
        assert body["authenticated"] and body["csrf_token"]
        sc = r.headers["set-cookie"].lower()
        assert "ks_session=" in sc and "httponly" in sc and "samesite=strict" in sc and "path=/ui/v1" in sc
        token = c.cookies.get("ks_session")
        assert token and token not in r.text and KEY not in r.text  # токен только в cookie
        s = c.get("/ui/v1/session")
        assert s.status_code == 200 and s.json()["csrf_token"] == body["csrf_token"]
        assert c.get("/ui/v1/jobs").status_code == 200
        # cookie не даёт доступа к API жюри
        assert c.get("/api/v1/jobs").status_code == 401
        # выход
        assert c.delete("/ui/v1/session").status_code == 200
        c.cookies.clear()
        assert c.get("/ui/v1/jobs").status_code == 401


def test_ui_password_login_and_csrf(client_factory):
    with client_factory(spec(ok_run), ui_password="s3cret-pass") as c:
        assert _login(c, password="bad").status_code == 401
        r = _login(c, password="s3cret-pass")
        assert r.status_code == 200
        csrf = r.json()["csrf_token"]
        files = {"file": ("X-1.mov", FTYP_ONLY, "video/quicktime")}
        assert c.post("/ui/v1/analyze", files=files).status_code == 403  # нет CSRF
        assert c.post("/ui/v1/analyze", files=files, headers={"X-CSRF-Token": "x" * 32}).status_code == 403
        bad_origin = {"X-CSRF-Token": csrf, "Origin": "https://evil.example"}
        assert c.post("/ui/v1/analyze", files=files, headers=bad_origin).status_code == 403
        ok = c.post("/ui/v1/analyze", files=files, headers={"X-CSRF-Token": csrf,
                                                           "Sec-Fetch-Site": "same-origin"})
        assert ok.status_code == 200, ok.text
        d = wait_job(c, ok.json()["job_id"], headers=None, prefix="/ui/v1")
        assert d["status"] == "done"


def test_ui_tampered_or_expired_cookie(client_factory):
    with client_factory(spec(ok_run)) as c:
        _login(c, api_key=KEY)
        tok = c.cookies.get("ks_session")
        c.cookies.clear()
        c.cookies.set("ks_session", tok[:-2] + ("AA" if not tok.endswith("AA") else "BB"), path="/ui/v1")
        assert c.get("/ui/v1/jobs").status_code == 401
    s = UiTokenSigner(b"x" * 48, ttl_s=10)
    t, exp = s.issue(now=1000)
    assert s.verify(t, now=1005) == exp
    assert s.verify(t, now=1011) is None
    assert UiTokenSigner(b"y" * 48, 10).verify(t, now=1005) is None
    assert s.check_csrf(t, s.csrf_for(t)) and not s.check_csrf(t, None)


def test_ui_login_rate_limited(client_factory):
    with client_factory(spec(ok_run), rl_ui_login="3/60") as c:
        codes = [_login(c, api_key="wrong").status_code for _ in range(5)]
        assert codes[:3] == [401, 401, 401] and codes[3:] == [429, 429]


def test_ui_can_be_disabled(client_factory):
    with client_factory(spec(ok_run), ui_enabled=False) as c:
        assert _login(c, api_key=KEY).status_code == 404


def test_keys_never_logged(client_factory, caplog):
    caplog.set_level(logging.DEBUG, logger="sentinel")
    with client_factory(spec(ok_run), ui_password="pw-in-log?") as c:
        logging.getLogger("sentinel").propagate = True
        upload(c, FTYP_ONLY)
        c.get("/api/v1/jobs", headers={"X-API-Key": "wrong-key-value"})
        _login(c, password="pw-in-log?")
        logging.getLogger("sentinel").propagate = False
    text = caplog.text + " ".join(str(r.__dict__) for r in caplog.records)
    for secret in (KEY, "wrong-key-value", "pw-in-log?"):
        assert secret not in text


def test_unit_helpers():
    a = ApiKeyAuth(["k1", "k2", ""])
    assert a.check("k2") and not a.check("k3") and not a.check(None) and len(a) == 2
    assert not ApiKeyAuth([]).check("anything")
    rl = RateLimiter()
    assert [rl.hit("b", "ip", 2, 60, now=0)[0] for _ in range(3)] == [True, True, False]
    assert rl.hit("b", "ip", 2, 60, now=31)[0]  # восстановление токенов
    assert client_ip("127.0.0.1", {"cf-connecting-ip": "1.2.3.4"}, ["127.0.0.1"]) == "1.2.3.4"
    assert client_ip("8.8.8.8", {"cf-connecting-ip": "1.2.3.4"}, ["127.0.0.1"]) == "8.8.8.8"
    assert cross_site({"sec-fetch-site": "cross-site"}, [])
    assert cross_site({"host": "a.com", "origin": "https://b.com"}, [])
    assert not cross_site({"host": "a.com", "origin": "https://a.com"}, [])
    assert not cross_site({"host": "localhost:8000", "origin": "http://localhost:5173"},
                          ["http://localhost:5173"])


def test_make_key_cli(capsys):
    from backend.api.make_key import main
    assert main(["--quiet"]) == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("ks_") and len(out) > 30


def test_deploy_env_aliases(tmp_path):
    """Имена переменных из deploy/ (A17) тоже понимаются."""
    from backend.api.config import Settings
    s = Settings.from_env(env={"SENTINEL_FRONTEND_DIST": str(tmp_path / "dist"),
                               "SENTINEL_DEADLINE_S": "240", "SENTINEL_API_KEY": "single",
                               "SENTINEL_API_KEYS": "a,b", "SENTINEL_DATA_DIR": str(tmp_path)})
    assert s.frontend_dir == tmp_path / "dist" and s.budget_s == 240.0
    assert s.api_keys == ["a", "b", "single"]
    s2 = Settings.from_env(env={"SENTINEL_DEADLINE_S": "240", "SENTINEL_BUDGET_S": "200"})
    assert s2.budget_s == 200.0
