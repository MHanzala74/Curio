import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "deploy" / "commit_state.sh"


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def make_repo(tmp_path):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(remote), str(work)], check=True, capture_output=True)
    git(work, "config", "user.email", "t@t")
    git(work, "config", "user.name", "t")
    (work / "README").write_text("x")
    git(work, "add", "README")
    git(work, "commit", "-qm", "init")
    git(work, "push", "-q", "-u", "origin", "HEAD")
    return remote, work


def run_script(work):
    return subprocess.run(["bash", str(SCRIPT)], cwd=work, capture_output=True, text=True)


def test_commits_state_even_when_profile_md_does_not_exist_yet(tmp_path):
    """Regression: a missing profile.md used to make `git add` stage nothing."""
    remote, work = make_repo(tmp_path)
    (work / "state").mkdir()
    (work / "state" / "live_state.json").write_text("{}")
    (work / "digest").mkdir()
    (work / "digest" / "2026-01-01.md").write_text("d")
    result = run_script(work)
    assert result.returncode == 0, result.stderr
    pushed = git(remote, "log", "--name-only", "--format=", "-1")
    assert "state/live_state.json" in pushed and "digest/2026-01-01.md" in pushed


def test_includes_profile_when_present_and_is_a_noop_when_unchanged(tmp_path):
    remote, work = make_repo(tmp_path)
    (work / "state").mkdir()
    (work / "state" / "live_state.json").write_text("{}")
    (work / "profile.md").write_text("p")
    assert run_script(work).returncode == 0
    assert "profile.md" in git(remote, "log", "--name-only", "--format=", "-1")
    before = git(remote, "rev-parse", "HEAD")
    again = run_script(work)
    assert again.returncode == 0 and "Nothing to commit" in again.stdout
    assert git(remote, "rev-parse", "HEAD") == before
