"""G4 domain-generalisation lineage guard (the G4 block of SERVER_TODO.md).

On the server the G4 arms inherit from the joint-training cross-entropy run `p3b_joint`
(MaxViT-L, softmax / CE head, joint_eyepacs=true; lineage p2a_plain -> p2b_none -> p3b_joint, preproc=none,
mix_prob=0, ema_decay=0). This module replays that lineage on the miniature synthetic project with the SAME
tags and the EXACT --set strings of SERVER_TODO.md, through main.py's own CLI entry point:

  p2b_none        reference (stand-in, phase2b_preprocess)
  p3b_joint       --inherit p2b_none --set joint_eyepacs=true          (stand-in, phase3b_joint)
  p6_amp          --inherit p3b_joint --set amp_mix=true amp_stage=pre amp_beta_max=0.10 winner_eligible=false
  p6_ms           --inherit p3b_joint --set mixstyle=true ms_p=0.5 ms_alpha=0.1 winner_eligible=false
  p6_amp_ms       --inherit p3b_joint --set amp_mix=true amp_stage=pre amp_beta_max=0.10 mixstyle=true
                                            winner_eligible=false
  p6_amp_nojoint  --inherit p2b_none  --set amp_mix=true amp_stage=pre amp_beta_max=0.10 winner_eligible=false
  main.py --mode phase_select --phase phase6_domain_gen --reference p3b_joint --set select_report_only=true
  tools/recalibration_suite.py --out_dir results/master/recal_dg --tags p6_amp,p6_ms,p6_amp_ms,p6_amp_nojoint,p3b_joint

The backbone is a tiny timm MaxViT (maxvit_nano_rw_256 at 64 px, passed to timm via img_size_arg) because
GradeNet's MixStyle path runs the MaxViT stage loop (stem / stages / norm / forward_head), which a ResNet cannot
exercise. Training-time behaviour is instrumented in-process (num_workers=0): MixStyle calls, the EyePACS/APTOS
batch layout that cross-domain MixStyle relies on, the amplitude-mix style bank and the amplitude-mix calls.
"""

import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from tests import synth

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHASE = 'phase6_domain_gen'
SETS = ['aptos_test', 'messidor_dev', 'messidor_test', 'ddr_dev', 'ddr_test', 'deepdrid_dev', 'deepdrid_test',
        'idrid_dev', 'idrid_test']
AMP = ['amp_mix=true', 'amp_stage=pre', 'amp_beta_max=0.10']
# tag -> (parent, --set tokens) exactly as in SERVER_TODO.md (G4)
G4_ARMS = {
    'p6_amp': ('p3b_joint', AMP + ['winner_eligible=false']),
    'p6_ms': ('p3b_joint', ['mixstyle=true', 'ms_p=0.5', 'ms_alpha=0.1', 'winner_eligible=false']),
    'p6_amp_ms': ('p3b_joint', AMP + ['mixstyle=true', 'winner_eligible=false']),
    'p6_amp_nojoint': ('p2b_none', AMP + ['winner_eligible=false']),
}
MS_ARMS = ('p6_ms', 'p6_amp_ms')
AMP_ARMS = ('p6_amp', 'p6_amp_ms', 'p6_amp_nojoint')
JOINT_ARMS = ('p6_amp', 'p6_ms', 'p6_amp_ms')
RECAL_TAGS = 'p6_amp,p6_ms,p6_amp_ms,p6_amp_nojoint,p3b_joint'


def _main(argv):
    """Run main.py's CLI entry point in-process (so the instrumentation below sees the training loop)."""
    import main as main_mod
    old = sys.argv
    sys.argv = ['main.py'] + list(argv)
    try:
        main_mod.main()
    finally:
        sys.argv = old


def _experiment(tag, phase, inherit=None, sets=()):
    argv = ['--mode', 'experiment', '--phase', phase, '--tag', tag]
    if inherit:
        argv += ['--inherit', inherit]
    if sets:
        argv += ['--set'] + list(sets)
    _main(argv)


@pytest.fixture(scope='module')
def g4(tmp_path_factory):
    """Synthetic project with a tiny MaxViT; trains reference -> joint CE -> the four G4 arms (instrumented)."""
    import src.training.trainer as trainer_mod
    import src.data.preprocess as preprocess_mod
    from src.models.mixstyle import MixStyle
    from src.data.samplers import JointBatchSampler

    root = tmp_path_factory.mktemp('g4proj')
    cwd = os.getcwd()
    os.chdir(root)
    rec = {'cur': None}

    def bucket():
        return rec.setdefault(rec['cur'], {'ms_train': 0, 'ms_mixed': 0, 'ms_eval': 0, 'amp_calls': 0,
                                           'banks': [], 'joint_batches': [], 'joint_n_a': []})

    orig_ms_forward = MixStyle.forward

    def ms_forward(self, x):
        out = orig_ms_forward(self, x)
        b = bucket()
        if self.training:
            b['ms_train'] += 1
            b['ms_mixed'] += int(out is not x)
        else:
            b['ms_eval'] += 1
        return out

    orig_bank = trainer_mod.build_style_bank

    def bank_spy(cfg, tr_df, logger=None):
        bank = orig_bank(cfg, tr_df, logger)
        bucket()['banks'].append(None if bank is None else list(bank))
        return bank

    orig_amp = preprocess_mod.amplitude_mix

    def amp_spy(*a, **kw):
        bucket()['amp_calls'] += 1
        return orig_amp(*a, **kw)

    orig_iter = JointBatchSampler.__iter__

    def iter_spy(self):
        b = bucket()
        b['joint_n_a'].append(self.n_a)
        for batch in orig_iter(self):
            b['joint_batches'].append((self.n_a, list(batch)))
            yield batch

    mp = pytest.MonkeyPatch()
    try:
        shutil.copytree(os.path.join(REPO, 'configs'), 'configs')
        cfg = synth.build(str(root))
        cfg['dedup_exclude_max'] = -1
        # tiny MaxViT; GradeNet forwards img_size_arg to timm.create_model(img_size=...)
        cfg.update({'backbone': 'maxvit_nano_rw_256', 'img_size_arg': 64})
        synth.install_config(cfg)

        from src.data.splits import make_splits
        from src.data.external import make_external_splits
        from tools.dedup_audit import run_dedup_audit, run_dedup_external
        from tests.test_pipeline import _cfg
        make_splits(cfg)
        run_dedup_audit(cfg)
        run_dedup_external(cfg)
        make_external_splits(_cfg('make_external_splits'))

        mp.setattr(MixStyle, 'forward', ms_forward)
        mp.setattr(trainer_mod, 'build_style_bank', bank_spy)
        mp.setattr(preprocess_mod, 'amplitude_mix', amp_spy)
        mp.setattr(JointBatchSampler, '__iter__', iter_spy)

        # server lineage stand-ins (p2b_none: preproc=none, CE, mix_prob=0, ema_decay=0 come from the synth cfg)
        for tag, phase, parent, sets in (
                ('p2b_none', 'phase2b_preprocess', None, ['loss=ce', 'preproc=none', 'head_type=softmax']),
                ('p3b_joint', 'phase3b_joint', 'p2b_none', ['joint_eyepacs=true'])):
            rec['cur'] = tag
            _experiment(tag, phase, parent, sets)
        for tag, (parent, sets) in G4_ARMS.items():
            rec['cur'] = tag
            _experiment(tag, PHASE, parent, sets)
        rec['cur'] = 'phase_select'
        summaries_before = {t: open(os.path.join('results', PHASE if t in G4_ARMS else 'phase3b_joint', t,
                                                 'summary.csv')).read() for t in list(G4_ARMS) + ['p3b_joint']}
        _main(['--mode', 'phase_select', '--phase', PHASE, '--reference', 'p3b_joint',
               '--set', 'select_report_only=true'])
        rec['cur'] = None
        mp.undo()
        yield {'root': root, 'rec': rec, 'summaries_before': summaries_before, 'eyepacs_dir': cfg['paths']['eyepacs']['images'],
               'aptos_dirs': [cfg['paths']['aptos']['train_images'], cfg['paths']['aptos']['val_images']]}
    finally:
        mp.undo()
        os.chdir(cwd)


@pytest.fixture
def at_g4(g4, monkeypatch):
    monkeypatch.chdir(g4['root'])
    return g4


def _snapshot(tag):
    return yaml.safe_load(open(os.path.join('checkpoints', tag, 'config_snapshot.yaml')))


def test_every_g4_arm_finishes_with_every_metric_on_every_set(at_g4):
    from src.evaluation.evaluate import SUMMARY_METRICS
    for tag in list(G4_ARMS) + ['p3b_joint']:
        d = os.path.join('results', 'phase3b_joint' if tag == 'p3b_joint' else PHASE, tag)
        assert os.path.isfile(os.path.join(d, 'summary.csv')), tag
        summ = pd.read_csv(os.path.join(d, 'summary.csv')).iloc[0]
        long = pd.read_csv(os.path.join(d, 'metrics_long.csv'))
        assert set(long['set']) == set(SETS) | {'oof'}, tag
        for s in SETS:
            got = set(long[(long['set'] == s) & (long['task'] != 'devcal')]['metric'])
            assert set(SUMMARY_METRICS) <= got, (tag, s, set(SUMMARY_METRICS) - got)
            for m in ('acc', 'qwk', 'f1_macro', 'bin_sens', 'bin_spec', 'bin_auroc'):
                assert f'{s}_{m}' in summ.index and np.isfinite(float(summ[f'{s}_{m}'])), (tag, s, m)
            P = np.load(os.path.join(d, f'predictions_{s}.npz'))['y_prob']
            assert np.all(np.isfinite(P)) and np.allclose(P.sum(1), 1.0, atol=1e-5), (tag, s)
        for f in ('oof.npz', 'config.yaml', 'predictions_external_dev.npz', 'predictions_external_test.npz'):
            assert os.path.isfile(os.path.join(d, f)), (tag, f)
        if tag != 'p3b_joint':
            assert str(summ['winner_eligible']).lower() == 'false', tag
            for k in range(2):
                assert os.path.isfile(os.path.join('checkpoints', tag, f'fold{k}.pth')), (tag, k)


def test_g4_config_snapshots_carry_the_intended_flags(at_g4):
    joint = _snapshot('p3b_joint')
    assert joint['joint_eyepacs'] is True and joint['head_type'] == 'softmax' and joint['loss'] == 'ce'
    assert not joint['amp_mix'] and not joint['mixstyle']
    for tag, (parent, _) in G4_ARMS.items():
        for snap in (_snapshot(tag), yaml.safe_load(open(os.path.join('results', PHASE, tag, 'config.yaml')))):
            assert snap['inherit'] == parent, tag
            assert snap['head_type'] == 'softmax' and snap['loss'] == 'ce', tag
            assert snap['backbone'] == 'maxvit_nano_rw_256' and snap['preproc'] == 'none', tag
            assert snap['winner_eligible'] is False, tag
            assert snap['joint_eyepacs'] is (tag in JOINT_ARMS), tag
            assert not snap.get('extra_sources'), tag
            assert bool(snap['amp_mix']) is (tag in AMP_ARMS), tag
            assert bool(snap['mixstyle']) is (tag in MS_ARMS), tag
            if tag in AMP_ARMS:
                assert snap['amp_stage'] == 'pre' and snap['amp_beta_max'] == pytest.approx(0.10), tag
            if tag in MS_ARMS:
                assert snap['ms_p'] == pytest.approx(0.5) and snap['ms_alpha'] == pytest.approx(0.1), tag
                assert list(snap['ms_stages']) == [0, 1], tag
            # everything else is inherited unchanged from the parent
            par = _snapshot(parent)
            for key in ('lr', 'epochs', 'batch_size', 'mix_prob', 'ema_decay', 'img_size', 'img_size_arg', 'seed',
                        'n_folds', 'joint_w', 'eyepacs_max'):
                assert snap[key] == par[key], (tag, key)


def test_mixstyle_active_only_in_ms_arms_and_batches_are_cross_domain(at_g4):
    rec = at_g4['rec']
    for tag in list(G4_ARMS) + ['p2b_none', 'p3b_joint']:
        b = rec[tag]
        if tag in MS_ARMS:
            assert b['ms_train'] > 0 and b['ms_mixed'] > 0, (tag, b['ms_train'], b['ms_mixed'])
            assert b['ms_mixed'] < b['ms_train'], 'ms_p=0.5 must sometimes skip mixing'
            assert b['ms_eval'] > 0, 'evaluation forward passes go through the (identity) MixStyle modules'
        else:
            assert b['ms_train'] == b['ms_eval'] == 0, tag
        if tag in JOINT_ARMS or tag == 'p3b_joint':
            # MixStyle swaps the two batch halves: [APTOS | EyePACS] makes every pair cross-domain
            assert b['joint_batches'], tag
            for n_a, batch in b['joint_batches']:
                h = len(batch) // 2
                assert len(batch) % 2 == 0 and all(i < n_a for i in batch[:h]) and all(i >= n_a for i in batch[h:])
        else:
            assert not b['joint_batches'], tag

    from src.models.gradenet import GradeNet
    for tag in MS_ARMS:
        cfg = _snapshot(tag)
        cfg.pop('pretrain_path', None)
        m = GradeNet(cfg)
        assert m.ms_idx == [0, 1] and len(m.ms) == 2
        m.load_state_dict(torch.load(os.path.join('checkpoints', tag, 'fold0.pth'), map_location='cpu'))


def test_amplitude_mix_style_bank_is_built_and_used(at_g4):
    rec = at_g4['rec']
    eye = os.path.abspath(at_g4['eyepacs_dir'])
    aptos = [os.path.abspath(d) for d in at_g4['aptos_dirs']]
    for tag in list(G4_ARMS) + ['p3b_joint']:
        b = rec[tag]
        assert len(b['banks']) == 2, tag            # one bank per fold
        if tag not in AMP_ARMS:
            assert all(bank is None for bank in b['banks']) and b['amp_calls'] == 0, tag
            continue
        assert b['amp_calls'] > 0, tag
        for bank in b['banks']:
            assert bank, f'{tag}: empty style bank'
            assert all(os.path.isfile(p) for p in bank)
            dirs = {os.path.dirname(os.path.abspath(p)) for p in bank}
            if tag in JOINT_ARMS:
                assert dirs == {eye}, (tag, dirs)       # joint arms: EyePACS style bank
            else:
                assert dirs <= set(aptos), (tag, dirs)  # no-joint arm: APTOS training-fold style bank


def test_phase_select_report_only_leaves_winner_untouched(at_g4):
    pdir = os.path.join('results', PHASE)
    pt = pd.read_csv(os.path.join(pdir, 'phase_table_amended.csv'))
    assert set(G4_ARMS) <= set(pt['experiment'])
    assert not os.path.exists(os.path.join(pdir, 'WINNER.txt'))
    assert not os.path.exists(os.path.join(pdir, 'phase_table.csv'))
    for col in ('decision_original', 'decision_amended', 'd_oof_qwk', 'd_dev_mean_qwk', 'dev_sets_used'):
        assert col in pt.columns
    pt = pt.set_index('experiment')
    assert pt.loc['p6_amp_nojoint', 'dev_sets_used'] == 'messidor,ddr,deepdrid,idrid'
    for tag in G4_ARMS:
        assert str(pt.loc[tag, 'winner_eligible']).lower() == 'false'
    # report-only: no summary.csv (decision / is_winner) is rewritten, the reference's included
    for tag, before in at_g4['summaries_before'].items():
        d = os.path.join('results', PHASE if tag in G4_ARMS else 'phase3b_joint', tag)
        assert open(os.path.join(d, 'summary.csv')).read() == before, tag


def test_recalibration_suite_accepts_the_g4_arms(at_g4):
    out = os.path.join('results', 'master', 'recal_dg')
    r = subprocess.run([sys.executable, os.path.join(REPO, 'tools', 'recalibration_suite.py'),
                        '--out_dir', out, '--tags', RECAL_TAGS, '--ns', '5', '--draws', '2', '--B', '20',
                        '--B_pi', '10', '--misspec_draws', '2'],
                       capture_output=True, text=True, cwd=str(at_g4['root']))
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    s = pd.read_csv(os.path.join(out, 'recal_summary.csv'))
    assert set(RECAL_TAGS.split(',')) <= set(s['tag'])
    assert {'MEAN'} <= set(s['dataset'])
    for f in ('recal_draws.csv', 'recal_compare.csv', 'recal_notes.csv'):
        assert os.path.isfile(os.path.join(out, f))


def test_mixstyle_stage_loop_equals_timm_forward_in_eval():
    """With MixStyle inactive (eval) the manual stage loop must be the very network timm's forward computes."""
    from src.models.gradenet import GradeNet
    base = {'backbone': 'maxvit_nano_rw_256', 'img_size_arg': 64, 'pretrained': False, 'drop_path': 0.0}
    torch.manual_seed(0)
    a = GradeNet(dict(base, mixstyle=True, ms_p=1.0, ms_alpha=0.1)).eval()
    b = GradeNet(dict(base, mixstyle=False)).eval()
    b.load_state_dict(a.state_dict())
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-5)
        a.train()
        b.train()      # both in train mode: BatchNorm uses batch statistics in both, only MixStyle differs
        assert not torch.allclose(a.forward_features(x), b.forward_features(x), atol=1e-4), \
            'ms_p=1 in train mode must change the features'
