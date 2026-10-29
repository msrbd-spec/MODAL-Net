"""Shared pytest fixtures.

Several tests write into repo-relative output folders ("results/<phase>/<tag>",
"checkpoints/<tag>", "logs/<tag>", "splits/*.json"). Those are the SAME paths the
real experiments use, so a test run concurrent with a training run would clobber
real artifacts. Tests that write there request the isolated_cwd fixture so every
relative path resolves inside their own tmp directory.
"""
import os
import shutil
from pathlib import Path

import pytest


@pytest.fixture(scope='session')
def repo_root():
    """Absolute project root, captured once at session start (before any test chdirs away).

    The artefact guard anchors on this instead of the current working directory, so a test that
    runs inside its own tmp dir stays free while a cleanup that escapes it (absolute path, ../..,
    or a teardown running after the chdir was undone) is still recognised as a real artefact.
    """
    return Path.cwd().resolve()


@pytest.fixture
def isolated_cwd(tmp_path, monkeypatch):
    """Run the test with cwd = tmp_path, with empty output dirs available."""
    (tmp_path / 'configs').mkdir(exist_ok=True)
    src = Path('configs/config.yaml')
    if src.is_file():
        shutil.copy(src, tmp_path / 'configs' / 'config.yaml')
    for name in ('results', 'checkpoints', 'logs', 'splits', 'data'):
        (tmp_path / name).mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture(autouse=True)
def _protect_artifacts(request, monkeypatch, repo_root):
    """Hard guard: no test may delete anything inside the real results/checkpoints/splits/logs.

    A synthetic fixture that rmtree()d the repo-relative "results/<phase>" folder once wiped a
    complete real phase folder (checkpoints survived, but every oof.npz / predictions_*.npz /
    summary.csv / WINNER.txt of that phase was lost). Requesting `isolated_cwd` is deliberately
    NOT an exemption any more: deletions inside the test's own tmp dir resolve outside repo_root
    and stay allowed, while anything landing inside a real tree raises. Cleanups in isolated tests
    must therefore be anchored on the tmp path returned by isolated_cwd. os.rmdir is not
    intercepted (tests/test_zz_guard_check.py uses it to remove its probe folder).
    """
    protected = [repo_root / name for name in ('results', 'checkpoints', 'splits', 'logs')
                 if (repo_root / name).is_dir()]
    if not protected:
        return

    def _guard(target, action):
        tgt = Path(target).resolve()
        for prot in protected:
            if tgt == prot or prot in tgt.parents:
                raise RuntimeError(
                    f'BLOCKED: {request.node.name} tried to {action} {tgt}, which is inside the '
                    f'real {prot.name} tree. Run the test with the isolated_cwd fixture and anchor '
                    f'its cleanup paths on the temporary directory it returns.'
                )

    real_rmtree = shutil.rmtree

    def guarded_rmtree(path, *args, **kwargs):
        _guard(path, 'delete')
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(shutil, 'rmtree', guarded_rmtree)

    for _name in ('remove', 'unlink'):
        def guarded(path, *args, _real=getattr(os, _name), _n=_name, **kwargs):
            _guard(path, f'run os.{_n} on')
            return _real(path, *args, **kwargs)
        monkeypatch.setattr(os, _name, guarded)

    real_path_unlink = Path.unlink

    def guarded_path_unlink(self, *args, **kwargs):
        _guard(self, 'run Path.unlink on')
        return real_path_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'unlink', guarded_path_unlink)
