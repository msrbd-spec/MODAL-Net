"""Shared pytest fixtures.

Every test runs with its working directory set to its own temporary folder (autouse fixture
`_isolated_project`). Relative output paths used by the code ("results/<phase>/<tag>",
"checkpoints/<tag>", "logs/<tag>", "splits/*.json", "splits/external/*.csv") therefore always resolve
inside that folder, and a test run can never overwrite the real splits, results or checkpoints of the
project (an earlier version of the suite overwrote splits/messidor_*_ids.json and wrote synthetic
rows into results/master and results/tables).

`isolated_cwd` is kept as an alias for tests that want the temporary path explicitly.
"""
import os
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope='session')
def repo_root():
    """Absolute project root (the folder that contains main.py)."""
    return REPO_ROOT


@pytest.fixture(autouse=True)
def _isolated_project(tmp_path, monkeypatch):
    """cwd = a fresh temporary project with configs/ copied and empty output folders."""
    shutil.copytree(REPO_ROOT / 'configs', tmp_path / 'configs')
    for name in ('results', 'checkpoints', 'logs', 'splits', 'softlabels', 'data'):
        (tmp_path / name).mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def isolated_cwd(_isolated_project):
    return _isolated_project


@pytest.fixture(autouse=True)
def _protect_artifacts(request, monkeypatch, _isolated_project):
    """Hard guard: no test may delete anything inside the real results/checkpoints/splits/logs/softlabels."""
    protected = [REPO_ROOT / name for name in ('results', 'checkpoints', 'splits', 'logs', 'softlabels')
                 if (REPO_ROOT / name).is_dir()]

    def _guard(target, action):
        tgt = Path(target).resolve()
        for prot in protected:
            if tgt == prot or prot in tgt.parents:
                raise RuntimeError(
                    f'BLOCKED: {request.node.name} tried to {action} {tgt}, which is inside the '
                    f'real {prot.name} tree.')

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
