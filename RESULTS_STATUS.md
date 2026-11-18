(env) [imam.bokhary@agpu1156 DR]$ mkdir -p results/master/recal_oracleref
cp results/master/recal/fewshot_head_draws.csv results/master/recal_oracleref/
env/bin/python tools/recalibration_suite.py --tags p3b_joint,p10d_ce,p10q_ce,p4j_coralsrc_shared --extra_refs oracle_bayes --out_dir results/master/recal_oracleref --fig_dir results/master/recal_oracleref/figures 2>&1 | tee outputs/recal_oracleref.txt
env/bin/python - <<'EOF'
import pandas as pd
c = pd.read_csv('results/master/recal_oracleref/recal_compare.csv')
s = c[(c.reference == 'oracle_bayes') & (c.method == 'oracle_quantile') & (c.metric == 'qwk')
      & c.dataset.isin(['messidor', 'ddr', 'deepdrid', 'idrid', 'MEAN'])]
print(s[['tag', 'dataset', 'diff_point', 'diff_lo', 'diff_hi']].round(3).to_string(index=False))
k = ['tag', 'dataset', 'method', 'info', 'n']
a = pd.read_csv('results/master/recal/recal_summary.csv')
b = pd.read_csv('results/master/recal_oracleref/recal_summary.csv')
m = a.merge(b, on=k, suffixes=('_old', '_new'))
print('rows compared:', len(m), '| max |QWK change| vs R3:', float((m.qwk_mean_old - m.qwk_mean_new).abs().max()))
EOF
recalibration_suite: p3b_joint: 4 held-out sites analysed in 71.5 s (prior bootstrap 38.1 s)
recalibration_suite: p10d_ce: 3 held-out sites analysed in 39.2 s (prior bootstrap 13.6 s)
recalibration_suite: p10q_ce: 3 held-out sites analysed in 36.0 s (prior bootstrap 9.5 s)
recalibration_suite: p4j_coralsrc_shared: no target-adaptation predictions for bn/messidor, tent/messidor, bn/ddr, tent/ddr, bn/deepdrid, tent/deepdrid, bn/idrid, tent/idrid (bn_adapt / tent rows skipped)
recalibration_suite: p4j_coralsrc_shared: 4 held-out sites analysed in 40.5 s (prior bootstrap 7.4 s)
recalibration_suite: merged 800 rows of results/master/recal_oracleref/fewshot_head_draws.csv

MEAN external QWK (held-out TEST; per-tag held-out sets) by method:
tag                                                        p10d_ce  p10q_ce  p3b_joint  p4j_coralsrc_shared
info                    method                       n                                                     
none                    argmax                        0      0.799    0.766      0.788                0.804
                        default_cuts                  0      0.774    0.732      0.796                0.816
                        source_cuts                   0      0.811    0.782      0.787                0.815
unlabelled_target       upa                           0      0.770    0.727      0.772                0.785
                        mlls_bcts_bayes               0      0.625    0.378      0.512                0.632
                        mlls_bcts_quantile            0      0.683    0.442      0.559                0.661
                        em_raw_bayes                  0      0.565    0.464      0.560                0.772
                        em_raw_quantile               0      0.645    0.581      0.615                0.782
                        bbse_bayes                    0      0.742    0.588      0.680                0.701
                        bbse_quantile                 0      0.824    0.729      0.768                0.775
                        mlls_bcts_bagged_quantile     0      0.687    0.440      0.566                0.663
                        mlls_bcts_shrunk_quantile     0      0.692    0.442      0.566                0.665
                        bbse_bagged_quantile          0      0.825    0.728      0.768                0.776
                        bbse_shrunk_quantile          0      0.830    0.732      0.769                0.777
                        bn_adapt_argmax               0      0.801    0.778      0.815                  NaN
                        bn_adapt_default_cuts         0      0.808    0.735      0.790                  NaN
                        tent_argmax                   0      0.815    0.717      0.781                  NaN
                        tent_default_cuts             0      0.823    0.728      0.783                  NaN
prevalence_oracle       oracle_quantile               0      0.856    0.798      0.827                0.845
                        oracle_bayes                  0      0.792    0.742      0.771                0.780
prevalence_other_sites  other_sites_quantile          0      0.732    0.672      0.746                0.757
prevalence_misspecified misspecified_quantile_tv0.05  0      0.838    0.782      0.816                0.832
                        misspecified_quantile_tv0.10  0      0.818    0.748      0.789                0.805
n_labels                fewshot_plain                 100    0.837    0.787      0.820                0.838
                        fewshot_shrink                100    0.836    0.788      0.820                0.830
                        fewshot_prev_quantile         100    0.837    0.776      0.811                0.827
                        fewshot_offset                100    0.790    0.727      0.792                0.826
                        fewshot_temp                  100    0.799    0.766      0.788                0.804
                        fewshot_temp_cuts             100    0.783    0.764      0.796                0.809
                        fewshot_vector                100    0.805    0.777      0.802                0.817
all_dev_labels          all_dev_cuts                 -1      0.843    0.800      0.832                0.853
                        all_dev_temp                 -1      0.799    0.766      0.788                0.804
                        all_dev_temp_cuts            -1      0.781    0.767      0.792                0.810
n_labels                fewshot_head_ft               100    0.846    0.800      0.838                  NaN

MEAN_COMMON external QWK (datasets common to every tag: messidor,idrid) by method:
tag                                                        p10d_ce  p10q_ce  p3b_joint  p4j_coralsrc_shared
info                    method                       n                                                     
none                    argmax                        0      0.775    0.754      0.756                0.776
                        default_cuts                  0      0.745    0.713      0.759                0.790
                        source_cuts                   0      0.783    0.774      0.762                0.787
unlabelled_target       upa                           0      0.705    0.673      0.684                0.682
                        mlls_bcts_bayes               0      0.580    0.288      0.332                0.507
                        mlls_bcts_quantile            0      0.643    0.357      0.397                0.550
                        em_raw_bayes                  0      0.497    0.395      0.401                0.761
                        em_raw_quantile               0      0.597    0.558      0.495                0.779
                        bbse_bayes                    0      0.688    0.553      0.592                0.603
                        bbse_quantile                 0      0.793    0.714      0.722                0.739
                        mlls_bcts_bagged_quantile     0      0.643    0.355      0.412                0.554
                        mlls_bcts_shrunk_quantile     0      0.647    0.358      0.410                0.558
                        bbse_bagged_quantile          0      0.794    0.713      0.720                0.739
                        bbse_shrunk_quantile          0      0.802    0.719      0.723                0.742
                        bn_adapt_argmax               0      0.763    0.765      0.790                  NaN
                        bn_adapt_default_cuts         0      0.773    0.715      0.740                  NaN
                        tent_argmax                   0      0.777    0.739      0.780                  NaN
                        tent_default_cuts             0      0.782    0.732      0.766                  NaN
prevalence_oracle       oracle_quantile               0      0.826    0.782      0.793                0.803
                        oracle_bayes                  0      0.748    0.724      0.734                0.701
prevalence_other_sites  other_sites_quantile          0      0.644    0.595      0.631                0.634
prevalence_misspecified misspecified_quantile_tv0.05  0      0.802    0.766      0.781                0.789
                        misspecified_quantile_tv0.10  0      0.778    0.738      0.754                0.765
n_labels                fewshot_plain                 100    0.804    0.771      0.792                0.802
                        fewshot_shrink                100    0.804    0.771      0.789                0.806
                        fewshot_prev_quantile         100    0.801    0.765      0.778                0.787
                        fewshot_offset                100    0.758    0.711      0.752                0.795
                        fewshot_temp                  100    0.775    0.754      0.756                0.776
                        fewshot_temp_cuts             100    0.755    0.740      0.762                0.768
                        fewshot_vector                100    0.779    0.760      0.772                0.774
all_dev_labels          all_dev_cuts                 -1      0.813    0.788      0.802                0.818
                        all_dev_temp                 -1      0.775    0.754      0.756                0.776
                        all_dev_temp_cuts            -1      0.755    0.744      0.753                0.770
n_labels                fewshot_head_ft               100    0.818    0.790      0.813                  NaN

1 notes (NaN / skipped items) -> results/master/recal_oracleref/recal_notes.csv
written: results/master/recal_oracleref/recal_draws.csv, recal_summary.csv, recal_compare.csv, prevalence_estimates.csv, binary_operating_points.csv, ordinal_discrimination.csv, indist_control.csv, aptos_dup_sensitivity.csv, recal_fitted.csv, recal_notes.csv; results/master/recal_oracleref/figures/fig_recal_levels, fig_recal_misspec, fig_recal_operating_points (.png/.pdf)
timing: analyse p3b_joint 71.5 s; analyse p10d_ce 39.2 s; analyse p10q_ce 36.0 s; analyse p4j_coralsrc_shared 40.5 s; summarise 3.8 s; compare (incl. joint draws x TEST bootstrap) 34.6 s; other tables 0.8 s; figures 6.3 s; total 232.8 s
                tag  dataset  diff_point  diff_lo  diff_hi
          p3b_joint messidor       0.059    0.032    0.085
          p3b_joint      ddr       0.095    0.079    0.112
          p3b_joint deepdrid       0.011   -0.015    0.037
          p3b_joint    idrid       0.060    0.013    0.109
          p3b_joint     MEAN       0.056    0.040    0.072
            p10d_ce messidor       0.067    0.041    0.098
            p10d_ce deepdrid       0.037    0.014    0.061
            p10d_ce    idrid       0.088    0.040    0.148
            p10d_ce     MEAN       0.064    0.043    0.088
            p10q_ce messidor       0.068    0.037    0.099
            p10q_ce      ddr       0.055    0.042    0.069
            p10q_ce    idrid       0.047   -0.003    0.100
            p10q_ce     MEAN       0.057    0.037    0.077
p4j_coralsrc_shared messidor       0.112    0.089    0.137
p4j_coralsrc_shared      ddr       0.037    0.028    0.047
p4j_coralsrc_shared deepdrid       0.022    0.004    0.042
p4j_coralsrc_shared    idrid       0.092    0.036    0.159
p4j_coralsrc_shared     MEAN       0.066    0.050    0.084
rows compared: 1316 | max |QWK change| vs R3: 0.0
