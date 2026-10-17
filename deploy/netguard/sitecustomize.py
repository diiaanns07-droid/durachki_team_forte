"""Автоподключение netguard: PYTHONPATH=<repo>/deploy/netguard и SENTINEL_NETGUARD=block|log."""
import os

if (os.environ.get("SENTINEL_NETGUARD") or "").lower() in ("block", "log"):
    import sentinel_netguard

    sentinel_netguard.install()
