from pathlib import Path

from openeval.cache import PredictionCache
from openeval.utils import hash_prompt


def test_cache_roundtrip(tmp_path: Path):
    cache = PredictionCache(tmp_path)
    k = hash_prompt(["adapter", "prompt1"])
    assert cache.get(k) is None
    cache.set(k, "out1")
    assert cache.get(k) == "out1"
    cache.close()


def test_cache_ttl(tmp_path: Path):
    cache = PredictionCache(tmp_path)
    k = hash_prompt(["a", "p"])
    cache.set(k, "o")
    assert cache.get(k, ttl=0.0) is None  # immediate expiry
    assert cache.get(k, ttl=None) == "o"
    cache.close()
