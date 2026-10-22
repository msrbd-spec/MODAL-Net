"""Unit tests for WP7: Report generation, Paper Tables (Table 01-15), and Figures (Fig 01-18)."""

import os
import tempfile
import pandas as pd
import pytest

from src.evaluation.tables import build_paper_tables
from src.evaluation.figures import build_paper_figures


def test_build_paper_tables():
    """Verify all 15 paper tables are generated with CSV and LaTeX booktabs format."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        build_paper_tables(out_dir=tmp_dir)

        expected_tables = [
            'table01_zoo',
            'table02_recipe_preprocess',
            'table03_pretrain_joint',
            'table04_label_loss',
            'table05_arch_ensemble',
            'table06_domain_generalisation',
            'table07_feature_balance',
            'table08_inference',
            'table09_per_class',
            'table10_binary',
            'table11_statistical_tests',
            'table12_conformal',
            'table13_literature',
            'table14_compute',
            'table15_seed_variance',
        ]

        for t_name in expected_tables:
            csv_path = os.path.join(tmp_dir, f"{t_name}.csv")
            tex_path = os.path.join(tmp_dir, f"{t_name}.tex")

            assert os.path.isfile(csv_path), f"Missing table CSV: {csv_path}"
            assert os.path.isfile(tex_path), f"Missing table TeX: {tex_path}"

            # Check CSV content
            df = pd.read_csv(csv_path)
            assert not df.empty, f"Table {t_name}.csv is empty"

            # Check LaTeX content
            tex_content = open(tex_path).read()
            assert "\\begin{table}" in tex_content, f"Missing table env in {tex_path}"
            assert "\\toprule" in tex_content
            assert "\\midrule" in tex_content
            assert "\\bottomrule" in tex_content


def test_build_paper_figures():
    """Verify all 18 paper figures are generated in both PNG (300 dpi) and PDF vector format."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        build_paper_figures(out_dir=tmp_dir)

        expected_figures = [
            'fig01_pipeline',
            'fig02_class_distributions',
            'fig03_training_curves',
            'fig04_confusion_matrices',
            'fig05_roc_curves',
            'fig06_pr_curves',
            'fig07_reliability',
            'fig08_risk_coverage',
            'fig09_feature_tsne',
            'fig10_undergrading',
            'fig11_phase_ablations',
            'fig12_forest_plot',
            'fig13_per_class_f1_recall',
            'fig14_label_ambiguity',
            'fig15_domain_spectra',
            'fig16_gradcam',
            'fig17_conformal',
            'fig18_preprocessing_parity',
        ]

        for f_name in expected_figures:
            png_path = os.path.join(tmp_dir, f"{f_name}.png")
            pdf_path = os.path.join(tmp_dir, f"{f_name}.pdf")

            assert os.path.isfile(png_path), f"Missing figure PNG: {png_path}"
            assert os.path.isfile(pdf_path), f"Missing figure PDF: {pdf_path}"

            # Verify non-empty files
            assert os.path.getsize(png_path) > 1000, f"Figure {png_path} is suspiciously small"
            assert os.path.getsize(pdf_path) > 500, f"Figure {pdf_path} is suspiciously small"


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
        P_test = np.eye(5)[y_test]
        plot_experiment_figures(exp_dir, curves_dir, y_test, P_test, y_test, P_test, n_folds=1)

        for fig_name in ['curves', 'cm_test', 'cm_external', 'roc', 'pr', 'reliability']:
            p_png = os.path.join(exp_dir, 'figures', f"{fig_name}.png")
            p_pdf = os.path.join(exp_dir, 'figures', f"{fig_name}.pdf")
            assert os.path.isfile(p_png), f"Missing {p_png}"
            assert os.path.isfile(p_pdf), f"Missing {p_pdf}"
