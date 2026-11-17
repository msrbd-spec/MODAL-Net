(env) [imam.bokhary@agpu1171 DR]$ column -s, -t results/phase8_inference/p8_conformal_ce/conformal.csv
column -s, -t results/phase8_inference/p8_conformal/conformal.csv
dataset        target_coverage  qhat    empirical_coverage  mean_width  singleton_share  singleton_acc  abstention_referable  aurc  acc_at_cov70  acc_at_cov80  acc_at_cov90  temperature
aptos_test     90.00            0.9960  100.00              3.61        0.00             0.00           60.38                 0.03  96.09         92.49         90.58         1.8273
messidor_dev   90.00            0.9960  100.00              4.44        0.00             0.00           92.84                 0.20  79.10         74.91         73.25         1.8273
messidor_test  90.00            0.9960  99.86               4.42        0.00             0.00           91.68                 0.19  79.00         76.52         74.44         1.8273
ddr_dev        90.00            0.9960  99.75               4.29        0.00             0.00           87.36                 0.09  87.73         84.97         82.41         1.8273
ddr_test       90.00            0.9960  99.23               4.42        0.00             0.00           90.54                 0.18  77.54         73.81         70.65         1.8273
deepdrid_dev   90.00            0.9960  100.00              4.93        0.00             0.00           100.00                0.21  72.04         71.47         68.80         1.8273
deepdrid_test  90.00            0.9960  100.00              4.89        0.00             0.00           99.50                 0.21  73.12         70.53         69.64         1.8273
idrid_dev      90.00            0.9960  100.00              4.52        0.00             0.00           89.10                 0.23  65.74         63.64         63.17         1.8273
idrid_test     90.00            0.9960  100.00              4.83        0.00             0.00           97.09                 0.40  52.78         48.78         49.46         1.8273
dataset        target_coverage  qhat    empirical_coverage  mean_width  singleton_share  singleton_acc  abstention_referable  aurc  acc_at_cov70  acc_at_cov80  acc_at_cov90  temperature
aptos_test     90.00            1.0000  100.00              3.55        0.00             0.00           60.38                 0.03  96.09         92.83         90.27         0.8001
messidor_dev   90.00            1.0000  99.71               4.58        0.00             0.00           93.41                 0.18  81.56         80.29         76.75         0.8001
messidor_test  90.00            1.0000  99.86               4.57        0.00             0.00           92.62                 0.16  82.07         79.57         77.39         0.8001
ddr_dev        90.00            1.0000  99.63               4.21        0.00             0.00           88.43                 0.08  88.38         88.24         85.36         0.8001
ddr_test       90.00            1.0000  99.34               4.37        0.00             0.00           91.34                 0.14  81.15         79.73         77.46         0.8001
deepdrid_dev   90.00            1.0000  100.00              4.73        0.00             0.00           99.25                 0.21  75.63         75.24         72.14         0.8001
deepdrid_test  90.00            1.0000  100.00              4.66        0.00             0.00           99.00                 0.18  74.55         75.55         74.65         0.8001
idrid_dev      90.00            1.0000  100.00              4.47        0.00             0.00           89.10                 0.15  77.51         75.45         74.19         0.8001
idrid_test     90.00            1.0000  100.00              4.81        0.00             0.00           99.03                 0.25  70.83         64.63         62.37         0.8001
