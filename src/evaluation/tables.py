"""Selection logic (phase_select), master tables, seed analysis, and paper tables.

Nothing in this module invents numbers: every table is built from summary.csv / metrics_long.csv /
phase tables of real runs. A paper table whose inputs do not exist is skipped and listed with the
reason in results/tables/TABLES_README.txt.
"""

import os
import glob
import re
import yaml
import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_ind

from src.evaluation.cutpoints import qwk_from_pred
from src.evaluation.stats import holm
from src.utils.config import find_exp, resolve
from src.utils.io_utils import SUMMARY_COLUMNS, fmt2, normalize_zoo_row


PHASE_ORDER = [
    'phase1_zoo',
    'phase2a_recipe',
    'phase2b_preprocess',
    'phase3a_pretrain',
    'phase3b_joint',
    'phase3c_seeds',
    'phase3c_ssl',
    'phase3c_dose',
    'phase3c_arch',
    'phase4_label_loss',
    'phase4j_label_loss',
    'phase5_architecture',
    'phase6_domain_gen',
    'phase8_inference',
    'phase9_final',
    'phase10_multisource',
]

EXT_DATASETS_ORDER = ['messidor', 'ddr', 'deepdrid', 'idrid']

MASTER_TABLE_COLS = [
    'phase', 'experiment', 'description', 'protocol', 'ext_set', 'train_sources', 'heldout_sets', 'seed',
    'head_type', 'oof_qwk', 'oof_acc', 'oof_bin_auroc',
    'test_acc', 'test_qwk', 'test_f1_macro', 'test_adj_acc', 'test_bin_acc', 'test_bin_auroc', 'test_sens_at_spec90',
    'ext_acc', 'ext_qwk', 'ext_mild_recall', 'ext_pred_nodr_share', 'ext_bin_acc', 'ext_bin_auroc',
    'ddr_test_qwk', 'ddr_test_acc', 'ddr_test_bin_auroc', 'ddr_test_mild_recall',
    'deepdrid_test_qwk', 'deepdrid_test_acc', 'deepdrid_test_bin_auroc', 'deepdrid_test_mild_recall',
    'idrid_test_qwk', 'idrid_test_acc', 'idrid_test_bin_auroc', 'idrid_test_mild_recall',
    'ext_mean_test_qwk', 'ext_mean_test_acc', 'ext_mean_test_bin_auroc', 'ext_mean_test_mild_recall',
    'ext_mean_dev_qwk', 'ext_full1744_acc', 'ext_full1744_qwk',
    'ece_test', 'ece_ext', 'gpu_hours', 'decision', 'is_winner',
]


# ---------------------------------------------------------------------------
# LaTeX formatting helpers
# ---------------------------------------------------------------------------

def _to_latex_booktabs(df: pd.DataFrame, highlight_cols: list = None) -> str:
    """Format DataFrame as a LaTeX booktabs table with best in bold and second underlined."""
    if df.empty:
        return ""
    df_str = df.copy().astype(object).where(pd.notna(df), '').astype(str)
    if highlight_cols:
        for col in highlight_cols:
            if col not in df.columns:
                continue
            vals = pd.to_numeric(df[col], errors='coerce')
            valid_vals = vals.dropna()
            if len(valid_vals) < 2:
                continue
            lower_is_better = any(term in col.lower() for term in ['ece', 'loss', 'mae', 'gap', 'nodr'])
            sorted_unique = sorted(valid_vals.unique(), reverse=not lower_is_better)
            for idx in vals[vals == sorted_unique[0]].index:
                df_str.loc[idx, col] = f"\\textbf{{{df_str.loc[idx, col]}}}"
            if len(sorted_unique) >= 2:
                for idx in vals[vals == sorted_unique[1]].index:
                    df_str.loc[idx, col] = f"\\underline{{{df_str.loc[idx, col]}}}"
    cols = df_str.columns.tolist()
    header = " & ".join([str(c).replace('_', '\\_') for c in cols]) + " \\\\\n"
    rows = [" & ".join(str(v).replace('_', '\\_') if not str(v).startswith('\\') else str(v)
                       for v in r.values) + " \\\\\n" for _, r in df_str.iterrows()]
    col_align = "l" + "r" * (len(cols) - 1)
    return ("\\begin{table}[htbp]\n\\centering\n"
            f"\\begin{{tabular}}{{{col_align}}}\n\\toprule\n{header}\\midrule\n{''.join(rows)}"
            "\\bottomrule\n\\end{tabular}\n\\end{table}\n")


# ---------------------------------------------------------------------------
# Selection Guard
# ---------------------------------------------------------------------------

def selection_guard(requested_path: str):
    """Enforce rule: APTOS test set and every external TEST split must NEVER be used for selection."""
    lower = str(requested_path).lower()
    if 'test' in lower and ('oof' not in lower and 'dev' not in lower):
        raise PermissionError(
            f"SELECTION GUARD VIOLATION: Phase selection attempted to read test set predictions: {requested_path}"
        )


# ---------------------------------------------------------------------------
# Fast metrics for the paired bootstrap
# ---------------------------------------------------------------------------

def _qwk(y, P):
    return qwk_from_pred(y, P.argmax(axis=1))


def _auroc(y, P):
    yb = (y >= 2).astype(int)
    npos, nneg = int(yb.sum()), int((1 - yb).sum())
    if npos == 0 or nneg == 0:
        return 0.5
    r = rankdata(P[:, 2:].sum(axis=1))
    return float((r[yb == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def _mild(y, P):
    m = (y == 1)
    return float((P.argmax(axis=1)[m] == 1).mean()) if m.any() else 0.0


def _paired(y, Pa, Pb, fn, B, seed=0, alpha=0.05):
    """Paired image bootstrap of fn(arm) - fn(ref). Returns delta, (lo, hi), p_two_sided, draws."""
    rng = np.random.default_rng(seed)
    n = len(y)
    d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n)
        d[b] = fn(y[i], Pa[i]) - fn(y[i], Pb[i])
    delta = fn(y, Pa) - fn(y, Pb)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return float(delta), (float(lo), float(hi)), float(p), d


def _paired_multi(per_ds, fn, B, seed=0):
    """Paired bootstrap of the MEAN over datasets of fn, each dataset resampled independently.

    per_ds: list of (y, P_arm, P_ref). Returns delta, (lo, hi), p_two_sided.
    """
    rng = np.random.default_rng(seed)
    d = np.empty(B)
    for b in range(B):
        vals = []
        for y, Pa, Pb in per_ds:
            i = rng.integers(0, len(y), len(y))
            vals.append(fn(y[i], Pa[i]) - fn(y[i], Pb[i]))
        d[b] = np.mean(vals)
    delta = float(np.mean([fn(y, Pa) - fn(y, Pb) for y, Pa, Pb in per_ds]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return delta, (float(lo), float(hi)), float(p)


def _load_npz(path):
    selection_guard(path)
    with np.load(path, allow_pickle=True) as f:
        y = np.asarray(f['y_true'] if 'y_true' in f.files else f['y'], dtype=int)
        P = np.asarray(f['P'] if 'P' in f.files else f['y_prob'])
        ids = [str(i) for i in f['ids']]
    return ids, y, P


def _align(arm, ref, what, ref_tag, phase):
    """Match arm and reference predictions of one DEV set by image id -> (y, P_arm, P_ref)."""
    ids_a, ya, Pa = arm
    ids_r, yr, Pr = ref
    pos = {i: k for k, i in enumerate(ids_a)}
    keep = [k for k, i in enumerate(ids_r) if i in pos]
    if len(keep) != len(ids_r) or len(keep) != len(ids_a):
        print(f'phase_select {phase}: {what}: DEV ids differ from {ref_tag} -> comparing the {len(keep)} common images')
    if not keep:
        raise ValueError(f'phase_select {phase}: {what} has no DEV image in common with {ref_tag}')
    idx_a = np.array([pos[ids_r[k]] for k in keep], dtype=int)
    idx_r = np.array(keep, dtype=int)
    assert np.array_equal(ya[idx_a], yr[idx_r]), f'{what}: DEV labels mismatch vs {ref_tag}'
    return yr[idx_r], Pa[idx_a], Pr[idx_r]


def _dev_file(exp_dir, ds):
    p = os.path.join(exp_dir, f'predictions_{ds}_dev.npz')
    if os.path.isfile(p):
        return p
    if ds == 'messidor':
        legacy = os.path.join(exp_dir, 'predictions_external_dev.npz')
        if os.path.isfile(legacy):
            return legacy
    return None


def _train_sources_of(summary_row, exp_dir):
    ts = str(summary_row.get('train_sources', '') or '')
    if ts and ts != 'nan':
        return set(ts.split(','))
    cfg_p = os.path.join(exp_dir, 'config.yaml')
    if os.path.isfile(cfg_p):
        from src.utils.config import train_sources
        return set(train_sources(yaml.safe_load(open(cfg_p)) or {}))
    return {'aptos'}


# ---------------------------------------------------------------------------
# phase_select
# ---------------------------------------------------------------------------

def phase_select(phase: str, reference: str, design: str = 'parallel', cfg: dict = None):
    """Compare all experiments of a phase against a reference and select the winner.

    Two endpoints are always computed and both decisions are written to the phase table:

    original (pre-registered): ADOPT iff the paired-OOF Delta QWK CI lower bound > 0 and the referable
        AUROC / OOF QWK guards hold (phase-specific rules for phase2a, phase2b, phase6 kept unchanged).
    amended: primary = mean over held-out external DEV sets (cfg select_dev_sets) of the QWK,
        paired bootstrap with every dataset resampled independently. ADOPT iff its CI lower bound > 0,
        Delta OOF QWK >= -noninf_margin (APTOS non-inferiority) and Delta mean-DEV referable AUROC
        >= -noninf_margin. In-domain differences are additionally tested for equivalence (TOST via the
        90 % CI inside +-tost_margin).

    cfg['select_endpoint'] picks the endpoint that sets `decision`, the winner and WINNER.txt.
    cfg['select_report_only'] = true writes phase_table_<endpoint>.csv only.
    """
    cfg = cfg or {}
    endpoint = cfg.get('select_endpoint', 'amended')
    report_only = bool(cfg.get('select_report_only', False))
    dev_sets = list(cfg.get('select_dev_sets') or EXT_DATASETS_ORDER)
    noninf = float(cfg.get('noninf_margin', 0.005))
    tost_margin = float(cfg.get('tost_margin', 0.01))

    ref_tag = resolve(reference)
    ref_dir = find_exp(ref_tag)
    ref_oof_path = os.path.join(ref_dir, 'oof.npz')
    if not os.path.isfile(ref_oof_path):
        raise FileNotFoundError(f"Reference OOF file not found: {ref_oof_path}")
    oof_ids_ref, y_oof_ref, P_oof_ref = _load_npz(ref_oof_path)
    ref_dev = {}
    for ds in dev_sets:
        p = _dev_file(ref_dir, ds)
        if p:
            ref_dev[ds] = _load_npz(p)
    if 'messidor' not in ref_dev and endpoint == 'original':
        raise FileNotFoundError(f"Reference Messidor DEV predictions not found in {ref_dir}")
    ref_summary = pd.read_csv(os.path.join(ref_dir, 'summary.csv')).iloc[0].to_dict()
    ref_sources = _train_sources_of(ref_summary, ref_dir)

    phase_dir = os.path.join('results', phase)
    exp_dirs = sorted(d for d in glob.glob(os.path.join(phase_dir, '*'))
                      if os.path.isdir(d) and os.path.isfile(os.path.join(d, 'summary.csv')))

    records, exp_summaries = [], {}
    for ed in exp_dirs:
        tag = os.path.basename(ed)
        summ = pd.read_csv(os.path.join(ed, 'summary.csv')).iloc[0].to_dict()
        exp_summaries[tag] = summ
        eligible = str(summ.get('winner_eligible', 'true')).lower() != 'false'
        if tag == ref_tag:
            records.append({'experiment': tag, 'is_ref': True, 'winner_eligible': eligible,
                            'oof_qwk_full': _qwk(y_oof_ref, P_oof_ref),
                            'dev_mean_qwk_full': float(np.mean([_qwk(v[1], v[2]) for d, v in ref_dev.items()
                                                                if d not in ref_sources] or [np.nan]))})
            continue
        oof_path = os.path.join(ed, 'oof.npz')
        if not os.path.isfile(oof_path):
            print(f'phase_select {phase}: {tag} has no oof.npz (not a trained arm) -> skipped')
            continue
        oof_ids, y_oof, P_oof = _load_npz(oof_path)
        assert np.array_equal(y_oof_ref, y_oof), f"OOF labels mismatch between {ref_tag} and {tag}"
        r = {'experiment': tag, 'is_ref': False, 'winner_eligible': eligible}
        r['d_oof_qwk'], (r['ci_lo'], r['ci_hi']), r['raw_p'], draws = _paired(y_oof, P_oof, P_oof_ref, _qwk, 10000)
        lo90, hi90 = np.percentile(draws, [5, 95])
        r['oof_tost_lo90'], r['oof_tost_hi90'] = float(lo90), float(hi90)
        r['oof_equivalent'] = bool(lo90 > -tost_margin and hi90 < tost_margin)
        r['d_bin_auroc'] = _paired(y_oof, P_oof, P_oof_ref, _auroc, 2000)[0]
        r['oof_qwk_full'] = _qwk(y_oof, P_oof)

        arm_dev = {}
        for ds in dev_sets:
            p = _dev_file(ed, ds)
            if p:
                arm_dev[ds] = _load_npz(p)
        # Messidor DEV deltas (used by the original phase2b / phase6 rules)
        if 'messidor' in arm_dev and 'messidor' in ref_dev:
            yd, Pd, Pr = _align(arm_dev['messidor'], ref_dev['messidor'], f'{tag}/messidor', ref_tag, phase)
            r['d_extdev_qwk'], (r['d_extdev_qwk_ci_lo'], _), _, _ = _paired(yd, Pd, Pr, _qwk, 2000)
            r['d_extdev_mild_recall'], (r['d_extdev_mild_ci_lo'], _), _, _ = _paired(yd, Pd, Pr, _mild, 2000)
        else:
            r.update({'d_extdev_qwk': np.nan, 'd_extdev_qwk_ci_lo': np.nan, 'd_extdev_mild_recall': np.nan,
                      'd_extdev_mild_ci_lo': np.nan})

        # Amended endpoint over held-out DEV sets common to reference and arm
        arm_sources = _train_sources_of(summ, ed)
        common = [ds for ds in dev_sets if ds not in arm_sources and ds not in ref_sources]
        if endpoint == 'amended':
            missing = [f'{who}:{ds}' for ds in common for who, dd in ((tag, arm_dev), (ref_tag, ref_dev))
                       if ds not in dd]
            if missing:
                raise FileNotFoundError(
                    f'phase_select {phase}: DEV predictions missing for {missing}. Run '
                    f'--mode evaluate (or evaluate_all) for these tags first, or restrict select_dev_sets.')
        common = [ds for ds in common if ds in arm_dev and ds in ref_dev]
        per_ds = [_align(arm_dev[ds], ref_dev[ds], f'{tag}/{ds}', ref_tag, phase) for ds in common]
        r['dev_sets_used'] = ','.join(common)
        if per_ds:
            r['d_dev_mean_qwk'], (r['d_dev_mean_qwk_ci_lo'], r['d_dev_mean_qwk_ci_hi']), r['dev_p'] = \
                _paired_multi(per_ds, _qwk, 2000)
            r['d_dev_mean_auroc'] = _paired_multi(per_ds, _auroc, 1000)[0]
            r['d_dev_mean_mild_recall'] = _paired_multi(per_ds, _mild, 1000)[0]
            r['dev_mean_qwk_full'] = float(np.mean([_qwk(y, Pa) for y, Pa, _ in per_ds]))
        else:
            r.update({'d_dev_mean_qwk': np.nan, 'd_dev_mean_qwk_ci_lo': np.nan, 'd_dev_mean_qwk_ci_hi': np.nan,
                      'dev_p': 1.0, 'd_dev_mean_auroc': np.nan, 'd_dev_mean_mild_recall': np.nan,
                      'dev_mean_qwk_full': np.nan})
        records.append(r)

    arms = [r for r in records if not r['is_ref']]
    if arms:
        for r, p in zip(arms, holm([r['raw_p'] for r in arms])):
            r['p_holm'] = float(p)
        for r, p in zip(arms, holm([r['dev_p'] for r in arms])):
            r['dev_p_holm'] = float(p)

    # Decisions under both endpoints
    for r in records:
        if r['is_ref']:
            r['decision_original'] = r['decision_amended'] = 'REFERENCE'
            continue
        # original rule (unchanged)
        if phase == 'phase6_domain_gen':
            orig = 'ADOPT' if (r['d_extdev_qwk'] > 0.0 and r['d_oof_qwk'] >= -0.005) else 'REJECT'
        elif phase == 'phase2b_preprocess':
            orig = 'ADOPT' if (r['d_extdev_qwk_ci_lo'] > 0.0 and r['d_extdev_qwk'] > 0.05
                               and r['d_extdev_mild_ci_lo'] > 0.0 and r['d_oof_qwk'] >= -0.005
                               and r['d_bin_auroc'] >= -0.005) else 'REJECT'
        elif r['ci_lo'] > 0.0 and r['d_bin_auroc'] >= -0.005 and r['d_oof_qwk'] >= -0.005:
            orig = 'ADOPT'
        elif phase == 'phase2a_recipe' and r['d_oof_qwk'] >= 0.005:
            orig = 'TENTATIVE'
        else:
            orig = 'REJECT'
        r['decision_original'] = orig
        lo = r.get('d_dev_mean_qwk_ci_lo')
        auc = r.get('d_dev_mean_auroc')
        amended_ok = (lo is not None and not np.isnan(lo) and lo > 0.0 and r['d_oof_qwk'] >= -noninf
                      and (auc is None or np.isnan(auc) or auc >= -noninf))
        amended_ok = amended_ok and r.get('dev_p_holm', 1.0) < 0.05   # Holm inside the phase (Part 1 S1)
        r['decision_amended'] = 'ADOPT' if amended_ok else 'REJECT'
    for r in records:
        r['decision'] = r[f'decision_{endpoint}']

    # Winner
    candidates = [r for r in records if r['decision'] in ('REFERENCE', 'ADOPT', 'TENTATIVE') and r['winner_eligible']]
    carried_reference = False
    if not candidates:
        candidates = [r for r in records if r['is_ref']]
    if not candidates:
        candidates = [{'experiment': ref_tag, 'is_ref': True, 'decision': 'REFERENCE',
                       'oof_qwk_full': _qwk(y_oof_ref, P_oof_ref),
                       'dev_mean_qwk_full': float(np.mean([_qwk(v[1], v[2]) for d, v in ref_dev.items()
                                                           if d not in ref_sources] or [np.nan]))}]
        carried_reference = True
    if endpoint == 'amended':
        # rank on the paired delta vs the reference (computed on the same held-out sets for both); reference = 0
        key = lambda x: (0.0 if x.get('is_ref') else np.nan_to_num(x.get('d_dev_mean_qwk', np.nan), nan=-9.0),
                         x.get('oof_qwk_full', 0.0))
    elif phase == 'phase6_domain_gen':
        valid = [c for c in candidates if c.get('d_oof_qwk', 0.0) >= -0.005] or candidates
        candidates = valid
        key = lambda x: (x.get('d_extdev_qwk', 0.0) if not x.get('is_ref') else 0.0, x.get('oof_qwk_full', 0.0))
    else:
        key = lambda x: (x.get('oof_qwk_full', 0.0), x.get('d_bin_auroc', 0.0) or 0.0)
    winner_record = sorted(candidates, key=key, reverse=True)[0]
    winner_tag = winner_record['experiment']

    # Phase tables
    rows = []
    fmt_keys = ['d_oof_qwk', 'ci_lo', 'ci_hi', 'p_holm', 'd_bin_auroc', 'oof_tost_lo90', 'oof_tost_hi90',
                'd_extdev_qwk', 'd_extdev_mild_recall', 'd_dev_mean_qwk', 'd_dev_mean_qwk_ci_lo',
                'd_dev_mean_qwk_ci_hi', 'dev_p_holm', 'd_dev_mean_auroc', 'd_dev_mean_mild_recall']
    for r in records:
        row = dict(exp_summaries.get(r['experiment'], {'experiment': r['experiment']}))
        row.update({k: fmt2(r.get(k)) if r.get(k) is not None else '' for k in fmt_keys})
        row.update({'experiment': r['experiment'], 'decision': r['decision'],
                    'decision_original': r['decision_original'], 'decision_amended': r['decision_amended'],
                    'endpoint': endpoint, 'oof_equivalent': r.get('oof_equivalent', ''),
                    'dev_sets_used': r.get('dev_sets_used', ''), 'winner': r['experiment'] == winner_tag})
        rows.append(row)
    phase_df = pd.DataFrame(rows)
    phase_df.to_csv(os.path.join(phase_dir, f'phase_table_{endpoint}.csv'), index=False)
    if report_only:
        print(f"phase_select {phase} ({endpoint}, report only): best arm {winner_tag}; "
              f"WINNER.txt and summaries untouched -> results/{phase}/phase_table_{endpoint}.csv")
        return winner_tag

    phase_df.to_csv(os.path.join(phase_dir, 'phase_table.csv'), index=False)
    tex_cols = [c for c in ['experiment', 'oof_qwk', 'd_oof_qwk', 'ci_lo', 'ci_hi', 'p_holm', 'd_dev_mean_qwk',
                            'd_dev_mean_qwk_ci_lo', 'decision_original', 'decision_amended', 'winner']
                if c in phase_df.columns]
    with open(os.path.join(phase_dir, 'phase_table.tex'), 'w') as f_tex:
        f_tex.write(_to_latex_booktabs(phase_df[tex_cols], highlight_cols=['oof_qwk', 'd_dev_mean_qwk']))
    for r in records:
        sp = os.path.join(find_exp(r['experiment']), 'summary.csv')
        sdf = pd.read_csv(sp)
        sdf['decision'] = str(r['decision'])
        sdf['is_winner'] = bool(r['experiment'] == winner_tag)
        sdf.to_csv(sp, index=False)
    reason = (f"Selected as phase winner ({endpoint} endpoint): decision={winner_record['decision']}, "
              f"OOF QWK={winner_record.get('oof_qwk_full', float('nan')):.4f}, "
              f"mean external DEV QWK={winner_record.get('dev_mean_qwk_full', float('nan')):.4f}."
              + (" No in-phase arm met the adoption rule; the reference is carried forward unchanged."
                 if carried_reference else ""))
    with open(os.path.join(phase_dir, 'WINNER.txt'), 'w') as f_win:
        f_win.write(f"{winner_tag}\n{reason}\n")
    print(f"phase_select for {phase} ({endpoint}): WINNER is {winner_tag}. "
          f"Table written to {os.path.join(phase_dir, 'phase_table.csv')}")
    return winner_tag


# ---------------------------------------------------------------------------
# master_table
# ---------------------------------------------------------------------------

def discover_phases():
    """PHASE_ORDER first, then every other results/phase* folder (sorted)."""
    found = sorted(os.path.basename(d) for d in glob.glob(os.path.join('results', 'phase*')) if os.path.isdir(d))
    return [p for p in PHASE_ORDER if p in found] + [p for p in found if p not in PHASE_ORDER]


def collect_summaries():
    rows = []
    for phase_name in discover_phases():
        for ed in sorted(glob.glob(os.path.join('results', phase_name, '*'))):
            sp = os.path.join(ed, 'summary.csv')
            if os.path.isdir(ed) and os.path.isfile(sp):
                df = pd.read_csv(sp)
                if not df.empty:
                    row = normalize_zoo_row(df.iloc[0].to_dict())
                    row['phase'] = phase_name
                    row['experiment'] = os.path.basename(ed)
                    rows.append(row)
    return rows


def master_table():
    """Compile results/master/* from every results/phase*/<tag>/ (summary.csv, metrics_long.csv, phase tables)."""
    master_dir = os.path.join('results', 'master')
    os.makedirs(master_dir, exist_ok=True)
    all_summaries = collect_summaries()
    if not all_summaries:
        print("master_table: No summaries found.")
        return
    full_df = pd.DataFrame(all_summaries)
    full_df.to_csv(os.path.join(master_dir, 'master_full.csv'), index=False)
    m_cols = [c for c in MASTER_TABLE_COLS if c in full_df.columns]
    master_df = full_df[m_cols]
    for c in MASTER_TABLE_COLS:
        if c not in master_df.columns:
            master_df = master_df.assign(**{c: ''})
    master_df = master_df[MASTER_TABLE_COLS]
    master_df.to_csv(os.path.join(master_dir, 'master_table.csv'), index=False)
    tex_cols = ['phase', 'experiment', 'oof_qwk', 'test_qwk', 'ext_qwk', 'ddr_test_qwk', 'deepdrid_test_qwk',
                'idrid_test_qwk', 'ext_mean_test_qwk', 'decision']
    with open(os.path.join(master_dir, 'master_table.tex'), 'w') as f_tex:
        f_tex.write(_to_latex_booktabs(master_df[tex_cols], highlight_cols=tex_cols[2:-1]))

    longs = []
    for r in all_summaries:
        p = os.path.join('results', r['phase'], r['experiment'], 'metrics_long.csv')
        if os.path.isfile(p):
            d = pd.read_csv(p)
            d['experiment'] = r['experiment']
            d['phase'] = r['phase']
            longs.append(d)
    if longs:
        pd.concat(longs, ignore_index=True).to_csv(os.path.join(master_dir, 'master_long.csv'), index=False)

    per_class_cols = ['phase', 'experiment'] + [c for c in full_df.columns
                                                if re.search(r'_(c\d|macro|mild_recall|nodr_share)$', c)]
    full_df[per_class_cols].to_csv(os.path.join(master_dir, 'master_per_class.csv'), index=False)
    binary_cols = ['phase', 'experiment'] + [c for c in full_df.columns if 'bin_' in c or 'sens_at_spec' in c]
    full_df[binary_cols].to_csv(os.path.join(master_dir, 'master_binary.csv'), index=False)

    config_rows = []
    for s in all_summaries:
        cfg_p = os.path.join('results', s['phase'], s['experiment'], 'config.yaml')
        if os.path.isfile(cfg_p):
            loaded_c = yaml.safe_load(open(cfg_p)) or {}
            config_rows.append({'experiment': s['experiment'], 'inherit': s.get('inherit', ''),
                                'set_overrides': s.get('set_overrides', ''),
                                **{f"cfg_{k}": v for k, v in loaded_c.items() if not isinstance(v, (dict, list))}})
    if config_rows:
        pd.DataFrame(config_rows).to_csv(os.path.join(master_dir, 'master_config.csv'), index=False)

    delta_rows = []
    for phase_name in discover_phases():
        pt_path = os.path.join('results', phase_name, 'phase_table.csv')
        if os.path.isfile(pt_path):
            for _, r in pd.read_csv(pt_path).iterrows():
                d = r.to_dict()
                d['phase'] = phase_name
                delta_rows.append(d)
    if delta_rows:
        pd.DataFrame(delta_rows).to_csv(os.path.join(master_dir, 'master_deltas.csv'), index=False)
    print(f"master_table: Compiled {len(all_summaries)} experiments into results/master/")


# ---------------------------------------------------------------------------
# seed_summary: seed replicates are found by configuration, not by tag name
# ---------------------------------------------------------------------------

# Keys that do not change what is trained (metadata, evaluation and reporting options).
_SIGNATURE_IGNORE = {
    'tag', 'phase', 'seed', 'set_overrides', 'inherit', 'base', 'description', 'members', 'kind', 'method',
    'ckpt_from', 'winner_eligible', 'eval_overwrite', 'evaluate_only', 'bootstrap_ci', 'fewshot_n',
    'fewshot_draws', 'dedup_exclude_max', 'external_sets', 'select_endpoint', 'select_report_only',
    'select_dev_sets', 'tost_margin', 'noninf_margin', 'seed_reference', 'report_final', 'report_baseline',
    'paths', 'num_workers', 'pretrained_fallback', 'grad_checkpointing', 'tta', 'tta_scales',
    'temperature_scaling', 'conformal_alpha', 'recalibrate', 'bundle_from',
}


def _signature_equal(a, b, defaults):
    keys = (set(a) | set(b)) - _SIGNATURE_IGNORE
    for k in keys:
        if k.startswith('xai_') or k.startswith('ssl_'):
            continue
        va, vb = a.get(k, defaults.get(k)), b.get(k, defaults.get(k))
        if va != vb:
            return False
    return True


SEED_METRICS = ['oof_qwk', 'oof_acc', 'test_acc', 'test_qwk', 'test_f1_macro', 'test_bin_auroc',
                'ext_mean_test_qwk', 'ext_mean_test_acc', 'ext_mean_test_bin_auroc', 'ext_mean_test_mild_recall'] + \
               [f'{d}_test_{m}' for d in EXT_DATASETS_ORDER for m in ('acc', 'qwk', 'f1_macro', 'bin_auroc',
                                                                       'bin_sens', 'bin_spec', 'mild_recall')]


def seed_groups():
    """Group every trained experiment with all experiments that differ only by seed / metadata."""
    defaults = yaml.safe_load(open('configs/config.yaml')) if os.path.isfile('configs/config.yaml') else {}
    items = []
    for s in collect_summaries():
        if s['phase'] == 'phase1_zoo':
            continue
        cp = os.path.join('results', s['phase'], s['experiment'], 'config.yaml')
        if not os.path.isfile(cp):
            continue
        c = yaml.safe_load(open(cp)) or {}
        sp = os.path.join('checkpoints', s['experiment'], 'config_snapshot.yaml')
        snap = (yaml.safe_load(open(sp)) or {}) if os.path.isfile(sp) else {}
        meta = {k: snap.get(k) or c.get(k) for k in ('members', 'kind', 'ckpt_from')}
        if meta['members'] or meta['kind'] == 'calibrate_conformal' or meta['ckpt_from']:
            continue  # ensembles / inference-only / conformal tags are not independent trainings
        items.append((s, c))
    groups = []
    for s, c in items:
        for g in groups:
            if _signature_equal(g['cfg'], c, defaults):
                g['members'].append((s, c))
                break
        else:
            groups.append({'cfg': c, 'members': [(s, c)]})
    out = []
    for g in groups:
        mem = sorted(g['members'], key=lambda sc: (int(sc[1].get('seed', 0) or 0), len(sc[0]['experiment'])))
        out.append({'group': mem[0][0]['experiment'], 'members': mem})
    return out


def seed_summary(phase: str = None, reference: str = None):
    """Mean +- SD over seed replicates for every configuration, and seed-level comparisons.

    Writes results/master/seed_variance.csv (group x metric: n, mean, sd, per-seed values) and
    results/master/seed_compare.csv (every group with >= 2 seeds vs the reference group: mean
    difference, Welch t-test p-value; seed is the unit of analysis). `phase` restricts the groups to
    those with at least one member in that phase.
    """
    groups = seed_groups()
    if phase:
        groups = [g for g in groups if any(s['phase'] == phase for s, _ in g['members'])]
    if not groups:
        print('seed_summary: no trained experiments found')
        return
    rows, values = [], {}
    for g in groups:
        tags = [s['experiment'] for s, _ in g['members']]
        seeds = [str(c.get('seed', 0)) for _, c in g['members']]
        for m in SEED_METRICS:
            vals = pd.to_numeric(pd.Series([s.get(m, np.nan) for s, _ in g['members']]), errors='coerce')
            v = vals.dropna().to_numpy(dtype=float)
            values[(g['group'], m)] = v
            if len(v) == 0:
                continue
            rows.append({'group': g['group'], 'members': ','.join(tags), 'seeds': ','.join(seeds), 'metric': m,
                         'n_seeds': len(v), 'mean': fmt2(v.mean()), 'sd': fmt2(v.std(ddof=1)) if len(v) > 1 else '',
                         'values': ','.join(fmt2(x) for x in v)})
    master_dir = os.path.join('results', 'master')
    os.makedirs(master_dir, exist_ok=True)
    pd.DataFrame(rows, columns=['group', 'members', 'seeds', 'metric', 'n_seeds', 'mean', 'sd', 'values']).to_csv(
        os.path.join(master_dir, 'seed_variance.csv'), index=False)

    ref = resolve(reference) if reference else None
    ref_group = next((g['group'] for g in groups if ref and any(s['experiment'] == ref for s, _ in g['members'])), None)
    cmp_rows = []
    if ref_group:
        for g in groups:
            if g['group'] == ref_group:
                continue
            for m in SEED_METRICS:
                a, b = values.get((g['group'], m), np.array([])), values.get((ref_group, m), np.array([]))
                if len(a) < 2 or len(b) < 2:
                    continue
                t = ttest_ind(a, b, equal_var=False)
                cmp_rows.append({'group': g['group'], 'reference': ref_group, 'metric': m, 'n_group': len(a),
                                 'n_reference': len(b), 'mean_group': fmt2(a.mean()), 'mean_reference': fmt2(b.mean()),
                                 'diff': fmt2(a.mean() - b.mean()), 'welch_p': fmt2(float(t.pvalue))})
    pd.DataFrame(cmp_rows, columns=['group', 'reference', 'metric', 'n_group', 'n_reference', 'mean_group',
                                    'mean_reference', 'diff', 'welch_p']).to_csv(
        os.path.join(master_dir, 'seed_compare.csv'), index=False)
    print(f'seed_summary: {len(groups)} configuration groups -> results/master/seed_variance.csv'
          + (f'; comparisons vs {ref_group} -> results/master/seed_compare.csv' if ref_group else
             '; no reference group with >= 2 seeds found (seed_compare.csv empty)'))


# ---------------------------------------------------------------------------
# build_paper_tables
# ---------------------------------------------------------------------------

def read_csv_safe(path):
    """pd.read_csv that returns an empty frame for a missing or zero-column file."""
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _fmt_ci(v, lo, hi, digits=2):
    if pd.isna(v):
        return ''
    if pd.isna(lo) or pd.isna(hi):
        return f'{v:.{digits}f}'
    return f'{v:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]'


def _model_sets_table(tags, metrics, label_map):
    rows = []
    for role, tag in tags:
        p = os.path.join(find_exp(tag), 'metrics_long.csv')
        d = pd.read_csv(p)
        d = d[d['task'] != 'devcal']
        for set_name in d['set'].drop_duplicates():
            sub = d[d['set'] == set_name].set_index('metric')
            row = {'model': f'{role} ({tag})', 'set': set_name, 'in_source': bool(sub['in_source'].iloc[0]),
                   'n': int(sub['n'].iloc[0])}
            for m in metrics:
                if m in sub.index:
                    r = sub.loc[m]
                    row[label_map.get(m, m)] = _fmt_ci(r['value'], r['ci_lo'], r['ci_hi'])
            rows.append(row)
    return pd.DataFrame(rows)


def build_paper_tables(master_df: pd.DataFrame = None, out_dir: str = 'results/tables', cfg: dict = None):
    """Build the paper tables from real results only (see TABLES_README.txt in out_dir for what was skipped)."""
    cfg = cfg or {}
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, 'table*.csv')) + glob.glob(os.path.join(out_dir, 'table*.tex')):
        os.remove(old)
    if master_df is None:
        master_table()
        mf = os.path.join('results', 'master', 'master_full.csv')
        master_df = pd.read_csv(mf) if os.path.isfile(mf) else pd.DataFrame()
    written, skipped = [], []

    def _save(df, name, highlight=None, note=None):
        if df is None or df.empty:
            skipped.append(f'{name}: no data')
            return
        df.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False)
        with open(os.path.join(out_dir, f'{name}.tex'), 'w') as f_tex:
            f_tex.write(_to_latex_booktabs(df, highlight_cols=highlight))
        written.append(name + (f'  ({note})' if note else ''))

    def _cols(df, mapping):
        out = pd.DataFrame()
        for src, dst in mapping.items():
            out[dst] = df[src].map(lambda x: fmt2(x) if isinstance(x, (int, float, np.floating)) and not
                                   isinstance(x, bool) else ('' if pd.isna(x) else str(x))) if src in df.columns else ''
        return out

    zoo = master_df[master_df['phase'] == 'phase1_zoo'] if not master_df.empty else pd.DataFrame()
    t1 = _cols(zoo, {'experiment': 'Model', 'test_acc': 'Test_Acc', 'test_qwk': 'Test_QWK', 'test_f1_macro': 'Test_F1',
                     'test_bin_acc': 'Test_Bin_Acc', 'ext_full1744_acc': 'Messidor1744_Acc',
                     'ext_full1744_qwk': 'Messidor1744_QWK', 'ext_full1744_bin_acc': 'Messidor1744_Bin_Acc'})
    _save(t1, 'table01_zoo_screening', note='single-fold models; external = all 1,744 Messidor-2 images, '
                                            'not comparable with the protocol runs')

    runs = master_df[master_df['phase'] != 'phase1_zoo'].copy() if not master_df.empty else pd.DataFrame()
    if not runs.empty:
        amended = {}
        for ph in runs['phase'].unique():
            pt = os.path.join('results', ph, 'phase_table_amended.csv')
            if os.path.isfile(pt):
                for _, r in pd.read_csv(pt).iterrows():
                    amended[r['experiment']] = r.get('decision_amended', '')
        runs = runs.copy()
        runs['decision_amended'] = runs['experiment'].map(amended).fillna('')
    t2 = _cols(runs, {'phase': 'Phase', 'experiment': 'Experiment', 'description': 'Description',
                      'train_sources': 'Train_sources', 'decision': 'Decision', 'decision_amended': 'Decision_amended',
                      'oof_qwk': 'OOF_QWK', 'test_qwk': 'APTOS_test_QWK', 'ext_qwk': 'Messidor_test_QWK',
                      'ddr_test_qwk': 'DDR_test_QWK', 'deepdrid_test_qwk': 'DeepDRiD_test_QWK',
                      'idrid_test_qwk': 'IDRiD_test_QWK', 'ext_mean_test_qwk': 'Ext_mean_QWK',
                      'ext_mean_test_bin_auroc': 'Ext_mean_AUROC', 'ext_mean_test_mild_recall': 'Ext_mean_Mild_recall'})
    _save(t2, 'table02_ablations', highlight=['OOF_QWK', 'Ext_mean_QWK', 'Ext_mean_AUROC'])

    final, base = cfg.get('report_final'), cfg.get('report_baseline')
    tags = [(r, resolve(t)) for r, t in (('final', final), ('baseline', base)) if t]
    if tags:
        lab5 = {'acc': 'Accuracy', 'qwk': 'QWK', 'prec_macro': 'Precision', 'rec_macro': 'Recall', 'f1_macro': 'F1',
                'mild_recall': 'Mild_recall'}
        labb = {'bin_acc': 'Accuracy', 'bin_prec': 'Precision', 'bin_sens': 'Sensitivity', 'bin_spec': 'Specificity',
                'bin_f1': 'F1', 'bin_auroc': 'AUROC', 'bin_auprc': 'AUPRC', 'sens_at_spec90': 'Sens_at_Spec90'}
        _save(_model_sets_table(tags, list(lab5), lab5), 'table03_main_5class',
              note='value [95 % bootstrap CI over images]')
        _save(_model_sets_table(tags, list(labb), labb), 'table04_main_binary',
              note='referable DR = grades 2-4; value [95 % CI]')
        pcs = []
        for role, tag in tags:
            pc = pd.read_csv(os.path.join(find_exp(tag), 'per_class.csv'))
            pc.insert(0, 'model', f'{role} ({tag})')
            pcs.append(pc)
        _save(pd.concat(pcs, ignore_index=True), 'table05_per_class')
        cps, fss = [], []
        for role, tag in tags:
            d = find_exp(tag)
            if os.path.isfile(os.path.join(d, 'cutpoints.csv')):
                c = read_csv_safe(os.path.join(d, 'cutpoints.csv'))
                if not c.empty:
                    c.insert(0, 'model', f'{role} ({tag})')
                    cps.append(c)
            if os.path.isfile(os.path.join(d, 'fewshot.csv')):
                f = read_csv_safe(os.path.join(d, 'fewshot.csv'))
                if not f.empty:
                    f.insert(0, 'model', f'{role} ({tag})')
                    fss.append(f)
        _save(pd.concat(cps, ignore_index=True).round(2) if cps else None, 'table07_cutpoints',
              note='argmax vs cut-points refitted on each dataset DEV split, evaluated on its TEST split')
        _save(pd.concat(fss, ignore_index=True).round(2) if fss else None, 'table08_fewshot')
        loc = [pd.read_csv(os.path.join(find_exp(t), 'xai', 'xai_localization.csv')).assign(model=f'{r} ({t})')
               for r, t in tags if os.path.isfile(os.path.join(find_exp(t), 'xai', 'xai_localization.csv'))]
        _save(pd.concat(loc, ignore_index=True) if loc else None, 'table12_xai_localization')
    else:
        for n in ('table03_main_5class', 'table04_main_binary', 'table05_per_class', 'table07_cutpoints',
                  'table08_fewshot', 'table12_xai_localization'):
            skipped.append(f'{n}: set --set report_final=<tag> report_baseline=<tag>')

    md = os.path.join('results', 'master', 'master_deltas.csv')
    if os.path.isfile(md):
        d = pd.read_csv(md)
        keep = [c for c in ['phase', 'experiment', 'decision_original', 'decision_amended', 'd_oof_qwk', 'ci_lo',
                            'ci_hi', 'p_holm', 'oof_equivalent', 'd_dev_mean_qwk', 'd_dev_mean_qwk_ci_lo',
                            'd_dev_mean_qwk_ci_hi', 'dev_p_holm', 'dev_sets_used'] if c in d.columns]
        _save(d[keep], 'table06_statistical_tests', note='paired image bootstrap vs the phase reference (seed 0)')
    else:
        skipped.append('table06_statistical_tests: no phase_table.csv yet (run phase_select)')

    _save(read_csv_safe(os.path.join('results', 'master', 'seed_variance.csv')), 'table09_seed_variance')
    _save(read_csv_safe(os.path.join('results', 'master', 'seed_compare.csv')), 'table09b_seed_compare')

    conf = []
    for p in sorted(glob.glob(os.path.join('results', 'phase*', '*', 'conformal.csv'))):
        c = read_csv_safe(p)
        c.insert(0, 'experiment', os.path.basename(os.path.dirname(p)))
        conf.append(c)
    _save(pd.concat(conf, ignore_index=True).round(2) if conf else None, 'table10_conformal')

    t11 = _cols(runs, {'phase': 'Phase', 'experiment': 'Experiment', 'n_params_M': 'Params_M',
                       'gpu_hours': 'GPU_hours', 'protocol': 'Protocol'})
    _save(t11, 'table11_compute', note='gpu_hours of tags rebuilt by tools/rebuild_oof.py time inference only')

    with open(os.path.join(out_dir, 'TABLES_README.txt'), 'w') as f:
        f.write('Paper tables built from real results only.\n\nWritten:\n')
        f.writelines(f'  {w}\n' for w in written)
        f.write('\nSkipped:\n')
        f.writelines(f'  {s}\n' for s in skipped)
        f.write('\nA literature comparison is not generated automatically: competitor numbers must come from '
                'verified papers (or be reproduced under this protocol) and be typed in by the authors.\n')
    print(f"build_paper_tables: {len(written)} tables written, {len(skipped)} skipped -> {out_dir}/TABLES_README.txt")
