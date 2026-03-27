# Cross-Complex Comparison

This summary aggregates the selected real-pair CPU runs already stored in `docs/training/`.
Canonical mode is deterministic once the complex manifest, model set, and seed set are fixed.

## Per-Complex Results

| Complex | Model | Seed | Schedule | Best Loss | Final Loss | Raw Ligand RMSE | Aligned Ligand RMSD | Training Seconds |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `10gs` | EGNN baseline | `43` | linear | `0.120972` | `1.290042` | `3.161712` | `2.914026` | `4.293` |
| `10gs` | EGNN baseline | `44` | linear | `0.148732` | `0.222952` | `3.072287` | `2.867563` | `2.129` |
| `10gs` | heterogeneous frame-based backbone | `43` | linear | `0.042226` | `0.173098` | `1.340780` | `1.278529` | `23.392` |
| `10gs` | heterogeneous frame-based backbone | `44` | linear | `0.042631` | `0.059739` | `1.546998` | `1.432156` | `18.673` |
| `11gs` | EGNN baseline | `43` | linear | `0.144927` | `1.357076` | `3.806530` | `2.843504` | `2.304` |
| `11gs` | EGNN baseline | `44` | linear | `0.135682` | `0.199490` | `3.375831` | `3.006598` | `2.067` |
| `11gs` | heterogeneous frame-based backbone | `43` | linear | `0.072647` | `0.719183` | `3.246566` | `2.601977` | `18.456` |
| `11gs` | heterogeneous frame-based backbone | `44` | linear | `0.065243` | `0.120691` | `2.307699` | `2.159142` | `18.224` |
| `13gs` | EGNN baseline | `43` | linear | `0.102298` | `2.325789` | `3.414255` | `2.866063` | `2.487` |
| `13gs` | EGNN baseline | `44` | linear | `0.134768` | `0.187123` | `3.298403` | `2.974045` | `2.161` |
| `13gs` | heterogeneous frame-based backbone | `43` | linear | `0.061275` | `0.136760` | `1.327952` | `1.216500` | `19.094` |
| `13gs` | heterogeneous frame-based backbone | `44` | linear | `0.060795` | `0.139933` | `1.531489` | `1.356931` | `18.193` |
| `16pk` | EGNN baseline | `43` | linear | `0.180369` | `1.800486` | `3.889992` | `3.169195` | `2.330` |
| `16pk` | EGNN baseline | `44` | linear | `0.148567` | `0.202970` | `4.337368` | `3.532266` | `2.160` |
| `16pk` | heterogeneous frame-based backbone | `43` | linear | `0.085922` | `0.302029` | `1.823879` | `1.742319` | `18.003` |
| `16pk` | heterogeneous frame-based backbone | `44` | linear | `0.102541` | `0.206404` | `2.343963` | `2.246991` | `18.267` |
| `184l` | EGNN baseline | `43` | linear | `0.116247` | `0.833495` | `1.919143` | `1.427510` | `2.688` |
| `184l` | EGNN baseline | `44` | linear | `0.135457` | `0.222395` | `1.966349` | `1.212138` | `2.264` |
| `184l` | heterogeneous frame-based backbone | `43` | linear | `0.016098` | `0.055196` | `1.234880` | `1.103838` | `20.229` |
| `184l` | heterogeneous frame-based backbone | `44` | linear | `0.015386` | `0.063734` | `1.165223` | `0.894109` | `19.058` |
| `185l` | EGNN baseline | `43` | linear | `0.102365` | `0.911278` | `1.652970` | `0.994488` | `2.633` |
| `185l` | EGNN baseline | `44` | linear | `0.115589` | `0.211303` | `1.618762` | `1.157095` | `2.340` |
| `185l` | heterogeneous frame-based backbone | `43` | linear | `0.004543` | `0.033049` | `1.021276` | `0.900424` | `19.393` |
| `185l` | heterogeneous frame-based backbone | `44` | linear | `0.003001` | `0.059530` | `1.243469` | `0.968223` | `18.785` |
| `186l` | EGNN baseline | `43` | linear | `0.091862` | `0.664741` | `1.683295` | `1.373423` | `2.639` |
| `186l` | EGNN baseline | `44` | linear | `0.084969` | `0.091869` | `3.104899` | `1.792791` | `2.304` |
| `186l` | heterogeneous frame-based backbone | `43` | linear | `0.010943` | `0.033518` | `1.125424` | `0.997668` | `18.458` |
| `186l` | heterogeneous frame-based backbone | `44` | linear | `0.009958` | `0.067720` | `1.171929` | `0.905040` | `18.779` |
| `187l` | EGNN baseline | `43` | linear | `0.085919` | `1.232727` | `2.325207` | `1.759051` | `2.325` |
| `187l` | EGNN baseline | `44` | linear | `0.092135` | `0.178017` | `1.885736` | `1.609203` | `2.311` |
| `187l` | heterogeneous frame-based backbone | `43` | linear | `0.009721` | `0.083400` | `1.136446` | `0.966146` | `19.467` |
| `187l` | heterogeneous frame-based backbone | `44` | linear | `0.005557` | `0.050871` | `1.249942` | `1.090997` | `18.824` |
| `188l` | EGNN baseline | `43` | linear | `0.100077` | `0.750414` | `1.601362` | `0.996103` | `2.394` |
| `188l` | EGNN baseline | `44` | linear | `0.122658` | `0.528734` | `3.964475` | `2.822306` | `2.237` |
| `188l` | heterogeneous frame-based backbone | `43` | linear | `0.006520` | `0.083881` | `1.093617` | `0.857318` | `19.069` |
| `188l` | heterogeneous frame-based backbone | `44` | linear | `0.002820` | `0.050159` | `1.260829` | `1.039004` | `18.559` |
| `1a07` | EGNN baseline | `43` | linear | `0.102424` | `1.856889` | `3.404836` | `2.774868` | `2.587` |
| `1a07` | EGNN baseline | `44` | linear | `0.176259` | `0.234308` | `3.380452` | `2.760517` | `2.133` |
| `1a07` | heterogeneous frame-based backbone | `43` | linear | `0.056005` | `0.246500` | `1.592110` | `1.535059` | `21.030` |
| `1a07` | heterogeneous frame-based backbone | `44` | linear | `0.062413` | `0.139338` | `1.927143` | `1.836245` | `27.158` |
| `1a08` | EGNN baseline | `43` | linear | `0.143170` | `1.897471` | `3.920114` | `3.077425` | `2.717` |
| `1a08` | EGNN baseline | `44` | linear | `0.162148` | `0.168486` | `3.583492` | `3.278188` | `2.110` |
| `1a08` | heterogeneous frame-based backbone | `43` | linear | `0.077290` | `0.554963` | `2.028960` | `1.985551` | `23.129` |
| `1a08` | heterogeneous frame-based backbone | `44` | linear | `0.084836` | `0.178141` | `2.278286` | `2.123695` | `18.082` |
| `1a09` | EGNN baseline | `43` | linear | `0.144273` | `1.521311` | `3.618550` | `3.153892` | `2.491` |
| `1a09` | EGNN baseline | `44` | linear | `0.156724` | `0.156724` | `3.708435` | `3.301512` | `2.035` |
| `1a09` | heterogeneous frame-based backbone | `43` | linear | `0.076204` | `0.508155` | `1.819182` | `1.780795` | `18.309` |
| `1a09` | heterogeneous frame-based backbone | `44` | linear | `0.082648` | `0.101729` | `2.297341` | `2.154404` | `18.588` |
| `1a0q` | EGNN baseline | `43` | linear | `0.102447` | `1.036762` | `2.897740` | `1.777422` | `2.342` |
| `1a0q` | EGNN baseline | `44` | linear | `0.153437` | `0.154933` | `2.937093` | `2.427365` | `2.153` |
| `1a0q` | heterogeneous frame-based backbone | `43` | linear | `0.025293` | `0.044566` | `1.198861` | `1.076961` | `18.249` |
| `1a0q` | heterogeneous frame-based backbone | `44` | linear | `0.025335` | `0.063543` | `1.301413` | `1.175420` | `18.031` |
| `1a0t` | EGNN baseline | `43` | linear | `0.135575` | `0.930837` | `2.673563` | `1.513977` | `2.296` |
| `1a0t` | EGNN baseline | `44` | linear | `0.143719` | `0.181559` | `2.546429` | `1.762889` | `2.079` |
| `1a0t` | heterogeneous frame-based backbone | `43` | linear | `0.024715` | `0.114113` | `1.189013` | `1.068826` | `18.323` |
| `1a0t` | heterogeneous frame-based backbone | `44` | linear | `0.021073` | `0.062539` | `1.700304` | `1.422166` | `18.015` |
| `1a1b` | EGNN baseline | `43` | linear | `0.141709` | `1.531226` | `3.895239` | `3.247977` | `2.212` |
| `1a1b` | EGNN baseline | `44` | linear | `0.240775` | `0.370742` | `4.546209` | `3.737448` | `2.021` |
| `1a1b` | heterogeneous frame-based backbone | `43` | linear | `0.073467` | `0.611377` | `2.332126` | `2.256415` | `17.900` |
| `1a1b` | heterogeneous frame-based backbone | `44` | linear | `0.123916` | `0.131568` | `2.565793` | `2.447539` | `17.974` |
| `1a1c` | EGNN baseline | `43` | linear | `0.144448` | `1.466212` | `3.841972` | `3.278188` | `2.215` |
| `1a1c` | EGNN baseline | `44` | linear | `0.211235` | `0.211235` | `3.717586` | `3.017229` | `2.118` |
| `1a1c` | heterogeneous frame-based backbone | `43` | linear | `0.084763` | `0.392703` | `2.163693` | `2.034874` | `18.571` |
| `1a1c` | heterogeneous frame-based backbone | `44` | linear | `0.114007` | `0.205842` | `2.341059` | `2.259366` | `23.073` |
| `1a1e` | EGNN baseline | `43` | linear | `0.152514` | `1.392815` | `3.841738` | `3.274412` | `2.225` |
| `1a1e` | EGNN baseline | `44` | linear | `0.215358` | `0.322615` | `4.103511` | `3.549736` | `2.122` |
| `1a1e` | heterogeneous frame-based backbone | `43` | linear | `0.074577` | `0.362235` | `2.092893` | `2.013097` | `18.047` |
| `1a1e` | heterogeneous frame-based backbone | `44` | linear | `0.091292` | `0.234991` | `2.121781` | `2.039440` | `18.078` |
| `1a28` | EGNN baseline | `43` | linear | `0.126399` | `1.090411` | `2.843255` | `2.287497` | `2.366` |
| `1a28` | EGNN baseline | `44` | linear | `0.149885` | `0.367476` | `3.410945` | `2.989787` | `2.265` |
| `1a28` | heterogeneous frame-based backbone | `43` | linear | `0.024863` | `0.041599` | `1.179139` | `1.090403` | `18.244` |
| `1a28` | heterogeneous frame-based backbone | `44` | linear | `0.019451` | `0.052153` | `1.228246` | `1.111798` | `19.440` |
| `1a2c` | EGNN baseline | `43` | linear | `0.159132` | `1.511745` | `2.984025` | `2.587238` | `2.276` |
| `1a2c` | EGNN baseline | `44` | linear | `0.164263` | `0.234273` | `3.592358` | `2.941659` | `2.129` |
| `1a2c` | heterogeneous frame-based backbone | `43` | linear | `0.074389` | `0.128274` | `1.162802` | `1.134483` | `18.096` |
| `1a2c` | heterogeneous frame-based backbone | `44` | linear | `0.071912` | `0.097463` | `1.619768` | `1.469226` | `18.019` |
| `1a30` | EGNN baseline | `43` | linear | `0.131299` | `0.903092` | `2.325467` | `2.158124` | `2.880` |
| `1a30` | EGNN baseline | `44` | linear | `0.115752` | `0.209922` | `2.706040` | `2.327859` | `2.113` |
| `1a30` | heterogeneous frame-based backbone | `43` | linear | `0.021655` | `0.045140` | `1.142733` | `1.021787` | `19.129` |
| `1a30` | heterogeneous frame-based backbone | `44` | linear | `0.018922` | `0.046560` | `1.244165` | `1.117241` | `20.432` |

## Aggregate Means

| Model | Schedule | Complexes | Runs | Mean Raw Ligand RMSE | Std Raw Ligand RMSE | Mean Aligned Ligand RMSD | Std Aligned Ligand RMSD | Success@2A | Success@5A | Mean Training Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EGNN baseline | - | `20` | `40` | `3.113941` | `0.799490` | `2.513564` | `0.780981` | `30.0%` | `100.0%` | `2.349` |
| heterogeneous frame-based backbone | - | `20` | `40` | `1.642479` | `0.529194` | `1.497803` | `0.517396` | `72.5%` | `100.0%` | `19.221` |

## Interpretation

- Across the selected comparison runs, the heterogeneous frame-based backbone reduces mean raw ligand RMSE by `47.3%` relative to the EGNN baseline.
- Across the same runs, the heterogeneous frame-based backbone reduces mean aligned ligand RMSD by `40.4%`.
- Success@2A changes from `30.0%` to `72.5%`.
- The improvement is not free: the heterogeneous frame-based backbone is roughly `8.18x` slower in mean training time.
- This is still a small-sample comparison, so it strengthens the project narrative but does not justify benchmark-scale claims.

## Source Logs

- `10gs` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/10gs_baseline_linear_seed43_log.md`
- `10gs` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/10gs_baseline_linear_seed44_log.md`
- `10gs` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/10gs_frame_backbone_linear_seed43_log.md`
- `10gs` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/10gs_frame_backbone_linear_seed44_log.md`
- `11gs` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/11gs_baseline_linear_seed43_log.md`
- `11gs` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/11gs_baseline_linear_seed44_log.md`
- `11gs` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/11gs_frame_backbone_linear_seed43_log.md`
- `11gs` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/11gs_frame_backbone_linear_seed44_log.md`
- `13gs` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/13gs_baseline_linear_seed43_log.md`
- `13gs` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/13gs_baseline_linear_seed44_log.md`
- `13gs` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/13gs_frame_backbone_linear_seed43_log.md`
- `13gs` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/13gs_frame_backbone_linear_seed44_log.md`
- `16pk` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/16pk_baseline_linear_seed43_log.md`
- `16pk` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/16pk_baseline_linear_seed44_log.md`
- `16pk` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/16pk_frame_backbone_linear_seed43_log.md`
- `16pk` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/16pk_frame_backbone_linear_seed44_log.md`
- `184l` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/184l_baseline_linear_seed43_log.md`
- `184l` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/184l_baseline_linear_seed44_log.md`
- `184l` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/184l_frame_backbone_linear_seed43_log.md`
- `184l` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/184l_frame_backbone_linear_seed44_log.md`
- `185l` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/185l_baseline_linear_seed43_log.md`
- `185l` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/185l_baseline_linear_seed44_log.md`
- `185l` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/185l_frame_backbone_linear_seed43_log.md`
- `185l` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/185l_frame_backbone_linear_seed44_log.md`
- `186l` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/186l_baseline_linear_seed43_log.md`
- `186l` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/186l_baseline_linear_seed44_log.md`
- `186l` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/186l_frame_backbone_linear_seed43_log.md`
- `186l` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/186l_frame_backbone_linear_seed44_log.md`
- `187l` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/187l_baseline_linear_seed43_log.md`
- `187l` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/187l_baseline_linear_seed44_log.md`
- `187l` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/187l_frame_backbone_linear_seed43_log.md`
- `187l` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/187l_frame_backbone_linear_seed44_log.md`
- `188l` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/188l_baseline_linear_seed43_log.md`
- `188l` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/188l_baseline_linear_seed44_log.md`
- `188l` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/188l_frame_backbone_linear_seed43_log.md`
- `188l` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/188l_frame_backbone_linear_seed44_log.md`
- `1a07` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/1a07_baseline_linear_seed43_log.md`
- `1a07` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/1a07_baseline_linear_seed44_log.md`
- `1a07` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/1a07_frame_backbone_linear_seed43_log.md`
- `1a07` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/1a07_frame_backbone_linear_seed44_log.md`
- `1a08` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/1a08_baseline_linear_seed43_log.md`
- `1a08` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/1a08_baseline_linear_seed44_log.md`
- `1a08` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/1a08_frame_backbone_linear_seed43_log.md`
- `1a08` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/1a08_frame_backbone_linear_seed44_log.md`
- `1a09` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/1a09_baseline_linear_seed43_log.md`
- `1a09` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/1a09_baseline_linear_seed44_log.md`
- `1a09` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/1a09_frame_backbone_linear_seed43_log.md`
- `1a09` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/1a09_frame_backbone_linear_seed44_log.md`
- `1a0q` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/1a0q_baseline_linear_seed43_log.md`
- `1a0q` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/1a0q_baseline_linear_seed44_log.md`
- `1a0q` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/1a0q_frame_backbone_linear_seed43_log.md`
- `1a0q` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/1a0q_frame_backbone_linear_seed44_log.md`
- `1a0t` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/1a0t_baseline_linear_seed43_log.md`
- `1a0t` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/1a0t_baseline_linear_seed44_log.md`
- `1a0t` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/1a0t_frame_backbone_linear_seed43_log.md`
- `1a0t` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/1a0t_frame_backbone_linear_seed44_log.md`
- `1a1b` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/1a1b_baseline_linear_seed43_log.md`
- `1a1b` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/1a1b_baseline_linear_seed44_log.md`
- `1a1b` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/1a1b_frame_backbone_linear_seed43_log.md`
- `1a1b` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/1a1b_frame_backbone_linear_seed44_log.md`
- `1a1c` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/1a1c_baseline_linear_seed43_log.md`
- `1a1c` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/1a1c_baseline_linear_seed44_log.md`
- `1a1c` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/1a1c_frame_backbone_linear_seed43_log.md`
- `1a1c` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/1a1c_frame_backbone_linear_seed44_log.md`
- `1a1e` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/1a1e_baseline_linear_seed43_log.md`
- `1a1e` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/1a1e_baseline_linear_seed44_log.md`
- `1a1e` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/1a1e_frame_backbone_linear_seed43_log.md`
- `1a1e` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/1a1e_frame_backbone_linear_seed44_log.md`
- `1a28` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/1a28_baseline_linear_seed43_log.md`
- `1a28` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/1a28_baseline_linear_seed44_log.md`
- `1a28` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/1a28_frame_backbone_linear_seed43_log.md`
- `1a28` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/1a28_frame_backbone_linear_seed44_log.md`
- `1a2c` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/1a2c_baseline_linear_seed43_log.md`
- `1a2c` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/1a2c_baseline_linear_seed44_log.md`
- `1a2c` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/1a2c_frame_backbone_linear_seed43_log.md`
- `1a2c` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/1a2c_frame_backbone_linear_seed44_log.md`
- `1a30` EGNN baseline seed `43` (linear): `docs/training/panel20/variance_mps/1a30_baseline_linear_seed43_log.md`
- `1a30` EGNN baseline seed `44` (linear): `docs/training/panel20/variance_mps/1a30_baseline_linear_seed44_log.md`
- `1a30` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/variance_mps/1a30_frame_backbone_linear_seed43_log.md`
- `1a30` heterogeneous frame-based backbone seed `44` (linear): `docs/training/panel20/variance_mps/1a30_frame_backbone_linear_seed44_log.md`
