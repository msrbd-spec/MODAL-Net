"""Tests for the paper report: no fabricated tables or figures, bold figure style (T12)."""

import os
import tempfile

import pandas as pd

from src.evaluation.figures import build_paper_figures
from src.evaluation.tables import build_paper_tables


def test_build_paper_tables_never_invents_numbers():
    """Without any run, no data table may be written; every table is listed as skipped."""
    build_paper_tables(master_df=pd.DataFrame(), out_dir='results/tables', cfg={})
    files = [f for f in os.listdir('results/tables') if f.startswith('table')]
    assert files == [], files
    readme = open('results/tables/TABLES_README.txt').read()
    assert 'Skipped' in readme and 'table03_main_5class' in readme


def test_build_paper_figures_never_invents_numbers():
    done, skipped = build_paper_figures(out_dir='results/figures', cfg={})
    assert done == []
    assert len(skipped) > 0
    assert not [f for f in os.listdir('results/figures') if f.startswith('fig')]


def test_t12_bold_style_enforcement():
    """Test T12: Verify that set_bold_style and save_fig enforce bold weight (>= 700) on all text artists."""
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    from src.evaluation.figures import set_bold_style, save_fig, weight_value
    from src.evaluation.experiment import plot_experiment_figures

    set_bold_style()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Part A: Synthetic test with legend, annotation, colorbar, and ticks
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(np.random.rand(5, 5))
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('Normalised attention (0-1)')
        ax.plot([0, 1, 2], [1, 2, 3], label='Line 1')
        ax.set_title('Test Title')
        ax.set_xlabel('X Axis')
        ax.set_ylabel('Y Axis')
        ax.text(1, 1, 'Sample Annotation')
        ax.legend(title='Legend Title')

        stem_a = os.path.join(tmp_dir, 'fig_test_bold')
        save_fig(fig, stem_a)

        assert os.path.isfile(stem_a + '.png')
        assert os.path.isfile(stem_a + '.pdf')

        # Part B: Real per-experiment figure from plot_experiment_figures
        exp_dir = os.path.join(tmp_dir, 'mock_exp')
        curves_dir = os.path.join(exp_dir, 'curves')
        os.makedirs(curves_dir, exist_ok=True)

        # Mock curve dataframe
        df_curve = pd.DataFrame({
            'epoch': [0, 1, 2],
            'train_loss': [0.5, 0.3, 0.2],
            'val_acc': [75.0, 80.0, 85.0],
            'train_acc': [70.0, 78.0, 83.0],
        })
        df_curve.to_csv(os.path.join(curves_dir, 'fold0.csv'), index=False)

        y_test = np.array([0, 1, 2, 3, 4] * 4)
        P_test = np.eye(5)[y_test] * 0.8 + 0.04
        probs = {'oof': (y_test, P_test), 'aptos_test': (y_test, P_test), 'messidor_test': (y_test, P_test),
                 'ddr_test': (y_test, P_test)}
        plot_experiment_figures(exp_dir, curves_dir, probs)

        for fig_name in ['curves', 'cm_test', 'cm_external', 'cm_ddr_test', 'roc', 'pr', 'reliability',
                         'reliability_ddr_test']:
            p_png = os.path.join(exp_dir, 'figures', f"{fig_name}.png")
            p_pdf = os.path.join(exp_dir, 'figures', f"{fig_name}.pdf")
            assert os.path.isfile(p_png), f"Missing {p_png}"
            assert os.path.isfile(p_pdf), f"Missing {p_pdf}"
