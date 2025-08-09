from openeval.utils import retry_call


def test_retry_succeeds_after_failures():
    calls = {"n": 0}

    def fn():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("boom")
        return 42

    out = retry_call(fn, retries=5, base_delay=0.0, jitter=0.0)
    assert out == 42
    assert calls["n"] == 3


def test_retry_exhausts():
    def fn():
        raise RuntimeError("x")

    try:
        retry_call(fn, retries=2, base_delay=0.0, jitter=0.0)
    except RuntimeError as e:
        assert "x" in str(e)
    else:
        assert False, "expected error"
