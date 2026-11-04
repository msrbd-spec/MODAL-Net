"""Self-check of the conftest.py safety net.

1. Every test runs in its own temporary project folder, so relative paths never reach the real tree.
2. The deletion guard blocks shutil.rmtree / os.remove / os.unlink / Path.unlink on any path inside the
   real results/, checkpoints/, splits/, logs/ or softlabels/ folders, given as an absolute path.

The probes below point at paths that do not exist: the guard raises before anything is touched, so these
tests never create or delete a file in the real project.
"""
import os
import shutil
from pathlib import Path

import pytest

PROTECTED = ('results', 'checkpoints', 'splits', 'logs', 'softlabels')
PROBE_NAME = '_pytest_guard_probe_does_not_exist'


def _real_protected(repo_root):
    found = [repo_root / n for n in PROTECTED if (repo_root / n).is_dir()]
    if not found:
        pytest.skip('no real results/checkpoints/splits/logs/softlabels folder in this checkout')
    return found


def test_tests_run_in_a_temporary_project(repo_root, tmp_path):
    cwd = Path(os.getcwd()).resolve()
    assert cwd == tmp_path.resolve()
    assert cwd != repo_root.resolve()
    assert Path('results').resolve().parent == cwd          # relative outputs stay inside tmp


def test_guard_blocks_rmtree_inside_real_results(repo_root):
    for prot in _real_protected(repo_root):
        with pytest.raises(RuntimeError, match='BLOCKED'):
            shutil.rmtree(str(prot / PROBE_NAME))
        with pytest.raises(RuntimeError, match='BLOCKED'):
            shutil.rmtree(str(prot))


def test_guard_blocks_file_deletion_inside_real_tree(repo_root):
    for prot in _real_protected(repo_root):
        probe = prot / f'{PROBE_NAME}.npz'
        for delete in (os.remove, os.unlink, Path.unlink):
            with pytest.raises(RuntimeError, match='BLOCKED'):
                delete(probe) if delete is Path.unlink else delete(str(probe))


def test_guard_allows_deleting_inside_the_tmp_project():
    os.makedirs(os.path.join('results', 'probe_dir'), exist_ok=True)
    with open(os.path.join('results', 'probe_dir', 'x.txt'), 'w') as fh:
        fh.write('x')
    os.remove(os.path.join('results', 'probe_dir', 'x.txt'))
    shutil.rmtree(os.path.join('results', 'probe_dir'))
    assert not os.path.exists(os.path.join('results', 'probe_dir'))
