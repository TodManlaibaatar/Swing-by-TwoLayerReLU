# E6_long_horizon

runs analysed: 20 (ReLU 10, linear 10); horizon split at t = 20, full horizon T = 100.

provenance: commits ['7ee5143ad9']; frozen files dirty in any run: False; long_horizon.py dirty in any run: False; devices ['NVIDIA A100-SXM4-40GB'].

## Pre-registered predictions

- **L1** -85 deg persistent on [0, 100] and never drops by more than 1e-6 after t = 20: PASS (10/10 ReLU seeds; largest drop 0.00e+00)
- **L2** at least 95% of the early -85 deg gain lost by t = 100: PASS (10/10; median [min, max] 99.1% [98.7, 99.3])
- **L3** no in-cone persistent direction: PASS (10/10 ReLU seeds with none)
- **L4** no linear direction with S >= 1e-5: PASS (10/10 linear seeds)
- **C1** E6 truncated at t = 20 vs E1 (20 matched runs): max |dS(-85)| = 0.00e+00, max |dE(-85, 20)| = 3.04e-29, max |dt_min| = 7.00e-02, max count difference (S >= 1e-4, in/off cone) = 0: CHECK

## Reported without prediction

| quantity (ReLU, -85 deg) | t = 20 | t = 40 | t = 60 | t = 100 |
|---|---|---|---|---|
| early gain lost, median [min, max] | 90.5% [89.5, 92.1] | 97.3% [96.6, 97.6] | 98.5% [97.8, 98.8] | 99.1% [98.7, 99.3] |
| E(-85 deg), median [min, max] | 0.49831 [0.49786, 0.49851] | 0.49948 [0.4993, 0.49953] | 0.49966 [0.4995, 0.49974] | 0.49979 [0.49968, 0.49983] |
| training loss (ReLU), median [min, max] | 0.000189 [0.000173, 0.000201] | 6.8e-05 [6.31e-05, 7.68e-05] | 3.85e-05 [3.62e-05, 4.68e-05] | 1.89e-05 [1.73e-05, 2.56e-05] |
| training loss (linear), median [min, max] | 1.93e-26 [1.9e-26, 1.95e-26] | 1.93e-26 [1.9e-26, 1.95e-26] | 1.93e-26 [1.9e-26, 1.95e-26] | 1.93e-26 [1.9e-26, 1.95e-26] |

- -85 deg minimum: E0 0.49995 [0.49994, 0.49996], E_min 0.48189 [0.47874, 0.48436], t_min 3.67 [3.62, 3.76]
- ReLU off-cone directions labelled persistent at 1e-4 (mean ± 95% CI over seeds): 112 ± 3.7 at T = 20, 124 ± 2.9 at T = 100
- ReLU in-cone directions labelled persistent: 0 in total at T = 20, 0 at T = 100; largest in-cone final excess at T = 100: 3.80e-09
- ReLU -85 deg label at T = 20: ['persistent']; at T = 100: ['persistent']

