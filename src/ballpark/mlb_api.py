"""Thin client for the public MLB Stats API."""
import requests

BASE = "https://statsapi.mlb.com/api/v1"


def get(path: str, **params) -> dict:
    r = requests.get(f"{BASE}/{path.lstrip('/')}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()
