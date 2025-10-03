from openeval.utils import get_project_root


def test_get_project_root_env_override(tmp_path, monkeypatch):
    # env var should win when set
    fake = tmp_path / "myroot"
    fake.mkdir()
    monkeypatch.setenv("OPENEVAL_PROJECT_ROOT", str(fake))
    assert get_project_root() == fake.resolve()


def test_get_project_root_detects_repo():
    root = get_project_root()
    # Expect pyproject or .git exists at or above
    assert (root / "pyproject.toml").exists() or (root / ".git").exists()
