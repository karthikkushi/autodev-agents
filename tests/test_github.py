"""GitHub push safety — Reject, the secret scan, private repos. Offline: gh is stubbed."""
import subprocess

import pytest

import agents.git_github as gg
from tools import secret_scan


def _repo(tmp_path, files: dict):
    root = tmp_path / "proj"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root)
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    _commit(root, "first")
    return root


def _commit(root, msg):
    subprocess.run(["git", "add", "-A"], cwd=root)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", msg], cwd=root)


def test_clean_repo_passes(tmp_path):
    assert secret_scan.scan_repo(str(_repo(tmp_path, {"app.py": "print(1)\n"}))) == []


def test_a_key_in_a_file_is_found_but_never_shown(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_SERVICE_API_KEY", "sekret-value-1234567890")
    found = secret_scan.scan_repo(str(_repo(tmp_path, {"src/app.py": 'KEY = "sekret-value-1234567890"\n'})))
    assert "src/app.py: the value of FAKE_SERVICE_API_KEY" in found
    assert not any("sekret-value" in f for f in found)


def test_a_key_deleted_later_is_still_found_in_history(tmp_path):
    root = _repo(tmp_path, {"config.js": 'const k = "AIza' + "B" * 35 + '";\n'})
    (root / "config.js").write_text("const k = process.env.KEY;\n")
    _commit(root, "remove key")
    assert secret_scan.scan_repo(str(root)) == ["git history: a Google API key"]


def test_a_tracked_env_file_is_found(tmp_path):
    assert ".env: an .env file is tracked" in secret_scan.scan_repo(str(_repo(tmp_path, {".env": "X=1\n"})))


def test_push_is_blocked_when_a_secret_is_found(tmp_path, monkeypatch):
    root = _repo(tmp_path, {"app.py": 'k = "AIza' + "C" * 35 + '"\n'})
    monkeypatch.setattr(gg, "_gh", lambda *a, **k: pytest.fail("must not reach GitHub"))
    monkeypatch.setattr(gg.control, "notify", lambda *a, **k: None)
    assert gg._push(str(root), "proj", {"agent_logs": []}) is False


def test_new_repos_are_created_private(tmp_path, monkeypatch):
    root = _repo(tmp_path, {"app.py": "print(1)\n"})
    calls = []

    def fake_gh(args, cwd, timeout=180):
        calls.append(args)
        if args[:2] == ["api", "user"]:
            return True, "someone"
        if args[:2] == ["repo", "view"]:
            return False, "not found"
        return True, ""
    monkeypatch.setattr(gg, "_gh", fake_gh)
    monkeypatch.setattr(gg.control, "notify", lambda *a, **k: None)
    assert gg._push(str(root), "my portfolio!", {"agent_logs": []}) is True
    create = next(c for c in calls if c[:2] == ["repo", "create"])
    assert create[2] == "someone/my-portfolio-" and "--private" in create and "--public" not in create


def test_not_logged_in_means_not_pushed(tmp_path, monkeypatch):
    root = _repo(tmp_path, {"app.py": "print(1)\n"})
    monkeypatch.setattr(gg, "_gh", lambda args, cwd, timeout=180: (False, "not logged in"))
    assert gg._push(str(root), "proj", {"agent_logs": []}) is False


@pytest.fixture
def flags(tmp_path, monkeypatch):
    for name in ("APPROVAL_FLAG", "REJECT_FLAG", "PENDING_FLAG"):
        monkeypatch.setattr(gg, name, tmp_path / name)
    monkeypatch.setattr(gg, "APPROVAL_POLL_INTERVAL", 0.01)
    return gg


def test_reject_ends_the_run_without_pushing(flags):
    from langgraph.graph import END
    from pipeline.graph import route_after_approval
    flags.PENDING_FLAG.write_text("github_push")
    flags.REJECT_FLAG.touch()
    state = flags.wait_for_github_approval({"agent_logs": []})
    assert state["github_approved"] is False
    assert not flags.PENDING_FLAG.exists() and not flags.REJECT_FLAG.exists()
    assert route_after_approval(state) == END


def test_approve_still_goes_to_the_push(flags):
    from pipeline.graph import route_after_approval
    flags.APPROVAL_FLAG.touch()
    state = flags.wait_for_github_approval({"agent_logs": []})
    assert state["github_approved"] is True and route_after_approval(state) == "github_push"


def test_commits_are_signed_as_the_github_account(tmp_path, monkeypatch):
    root = tmp_path / "p"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root)
    monkeypatch.setattr(gg, "_gh", lambda args, cwd, timeout=180: (True, "someone 12345"))
    gg._set_commit_author(str(root))
    get = lambda key: subprocess.run(["git", "config", key], cwd=root, capture_output=True, text=True).stdout.strip()
    assert get("user.name") == "someone"
    assert get("user.email") == "12345+someone@users.noreply.github.com"
    # Not logged in: git's own default is left alone.
    other = tmp_path / "q"
    other.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=other)
    monkeypatch.setattr(gg, "_gh", lambda args, cwd, timeout=180: (False, "not logged in"))
    gg._set_commit_author(str(other))
    assert subprocess.run(["git", "config", "--local", "user.name"], cwd=other,
                          capture_output=True, text=True).stdout.strip() == ""
