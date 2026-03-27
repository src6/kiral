# Cross-Complex Comparison

This summary aggregates the selected real-pair CPU runs already stored in `docs/training/`.
Canonical mode is deterministic once the complex manifest, model set, and seed set are fixed.

## Per-Complex Results

| Complex | Model | Seed | Schedule | Best Loss | Final Loss | Raw Ligand RMSE | Aligned Ligand RMSD | Training Seconds |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `10gs` | EGNN + complete frames | `42` | linear | `0.101763` | `0.358910` | `2.824258` | `2.391140` | `1.566` |
| `10gs` | EGNN + ligand context | `42` | linear | `0.074931` | `0.280431` | `3.108984` | `2.832278` | `1.378` |
| `10gs` | EGNN + typed edges | `42` | linear | `0.085730` | `0.187534` | `2.832145` | `2.575693` | `2.040` |
| `10gs` | EGNN baseline | `42` | linear | `0.096764` | `0.691036` | `3.616529` | `2.648997` | `1.350` |
| `10gs` | heterogeneous frame-based backbone | `42` | linear | `0.039260` | `0.061780` | `1.637889` | `1.420225` | `3.223` |
| `11gs` | EGNN + complete frames | `42` | linear | `0.105903` | `1.092911` | `3.357635` | `2.395390` | `1.685` |
| `11gs` | EGNN + ligand context | `42` | linear | `0.055449` | `0.621299` | `3.516052` | `3.002803` | `1.403` |
| `11gs` | EGNN + typed edges | `42` | linear | `0.120556` | `0.504845` | `3.216958` | `3.109054` | `1.980` |
| `11gs` | EGNN baseline | `42` | linear | `0.039134` | `0.045522` | `3.213288` | `2.960820` | `1.271` |
| `11gs` | heterogeneous frame-based backbone | `42` | linear | `0.061123` | `0.682064` | `1.849536` | `1.738707` | `3.408` |
| `13gs` | EGNN + complete frames | `42` | linear | `0.094373` | `0.195532` | `3.502167` | `2.471403` | `1.474` |
| `13gs` | EGNN + ligand context | `42` | linear | `0.084324` | `0.149778` | `4.085912` | `3.524910` | `1.404` |
| `13gs` | EGNN + typed edges | `42` | linear | `0.103542` | `0.851544` | `3.495441` | `2.516933` | `1.977` |
| `13gs` | EGNN baseline | `42` | linear | `0.080822` | `0.107243` | `4.452593` | `3.240630` | `1.288` |
| `13gs` | heterogeneous frame-based backbone | `42` | linear | `0.053906` | `0.055329` | `1.533526` | `1.513411` | `2.859` |
| `16pk` | EGNN + complete frames | `42` | linear | `0.104346` | `0.714885` | `3.444094` | `2.865223` | `1.504` |
| `16pk` | EGNN + ligand context | `42` | linear | `0.076596` | `1.829181` | `3.954975` | `3.559919` | `1.433` |
| `16pk` | EGNN + typed edges | `42` | linear | `0.110820` | `0.325460` | `3.938596` | `3.709288` | `2.135` |
| `16pk` | EGNN baseline | `42` | linear | `0.115222` | `0.648591` | `4.935036` | `4.076063` | `1.299` |
| `16pk` | heterogeneous frame-based backbone | `42` | linear | `0.095388` | `0.703123` | `2.310936` | `2.240291` | `3.213` |
| `184l` | EGNN + complete frames | `42` | linear | `0.106249` | `0.169903` | `2.417547` | `0.976391` | `1.958` |
| `184l` | EGNN + ligand context | `42` | linear | `0.088164` | `0.088164` | `1.826342` | `1.304355` | `1.788` |
| `184l` | EGNN + typed edges | `42` | linear | `0.090713` | `0.522577` | `1.480848` | `1.139301` | `2.445` |
| `184l` | EGNN baseline | `42` | linear | `0.094761` | `1.262720` | `1.871435` | `1.157649` | `1.652` |
| `184l` | heterogeneous frame-based backbone | `42` | linear | `0.013456` | `0.029134` | `1.299084` | `1.147352` | `4.382` |
| `185l` | EGNN + complete frames | `42` | linear | `0.074117` | `0.074117` | `1.499137` | `0.960903` | `2.070` |
| `185l` | EGNN + ligand context | `42` | linear | `0.075382` | `0.088558` | `1.314816` | `0.750050` | `1.957` |
| `185l` | EGNN + typed edges | `42` | linear | `0.088051` | `0.179155` | `1.494936` | `1.098040` | `2.629` |
| `185l` | EGNN baseline | `42` | linear | `0.069083` | `0.470628` | `2.536903` | `1.480046` | `1.733` |
| `185l` | heterogeneous frame-based backbone | `42` | linear | `0.011879` | `0.038537` | `1.204362` | `1.085284` | `3.960` |
| `186l` | EGNN + complete frames | `42` | linear | `0.093555` | `0.120306` | `2.000685` | `1.485593` | `2.115` |
| `186l` | EGNN + ligand context | `42` | linear | `0.090600` | `0.278270` | `1.838111` | `1.466929` | `1.904` |
| `186l` | EGNN + typed edges | `42` | linear | `0.074335` | `0.096314` | `2.003347` | `1.085879` | `2.642` |
| `186l` | EGNN baseline | `42` | linear | `0.076203` | `0.943669` | `2.136646` | `1.252245` | `1.727` |
| `186l` | heterogeneous frame-based backbone | `42` | linear | `0.005295` | `0.005295` | `0.980356` | `0.801763` | `3.942` |
| `187l` | EGNN + complete frames | `42` | linear | `0.077675` | `0.077675` | `1.282227` | `0.991376` | `2.175` |
| `187l` | EGNN + ligand context | `42` | linear | `0.079435` | `0.079435` | `1.562140` | `1.024548` | `1.885` |
| `187l` | EGNN + typed edges | `42` | linear | `0.092956` | `0.228719` | `1.684469` | `1.256721` | `2.569` |
| `187l` | EGNN baseline | `42` | linear | `0.078419` | `0.502475` | `1.842474` | `0.736532` | `1.799` |
| `187l` | heterogeneous frame-based backbone | `42` | linear | `0.013376` | `0.167010` | `0.997365` | `0.844008` | `3.949` |
| `188l` | EGNN + complete frames | `42` | linear | `0.077428` | `0.077428` | `1.849152` | `0.806250` | `2.114` |
| `188l` | EGNN + ligand context | `42` | linear | `0.068501` | `0.255283` | `1.506873` | `1.113386` | `1.823` |
| `188l` | EGNN + typed edges | `42` | linear | `0.110040` | `47.418266` | `1.705200` | `0.994540` | `2.563` |
| `188l` | EGNN baseline | `42` | linear | `0.070426` | `0.095945` | `2.147055` | `1.281683` | `1.715` |
| `188l` | heterogeneous frame-based backbone | `42` | linear | `0.012525` | `0.173573` | `1.276215` | `0.632821` | `3.757` |
| `1a07` | EGNN + complete frames | `42` | linear | `0.124448` | `0.209116` | `3.169263` | `2.626184` | `1.288` |
| `1a07` | EGNN + ligand context | `42` | linear | `0.080405` | `0.112299` | `3.218256` | `2.614156` | `1.231` |
| `1a07` | EGNN + typed edges | `42` | linear | `0.110089` | `1.070175` | `2.999341` | `2.702851` | `1.691` |
| `1a07` | EGNN baseline | `42` | linear | `0.098045` | `0.579624` | `3.217619` | `2.650544` | `1.156` |
| `1a07` | heterogeneous frame-based backbone | `42` | linear | `0.065348` | `0.091607` | `1.680381` | `1.523774` | `2.839` |
| `1a08` | EGNN + complete frames | `42` | linear | `0.136532` | `0.527549` | `3.037026` | `2.660791` | `1.375` |
| `1a08` | EGNN + ligand context | `42` | linear | `0.074234` | `2.284801` | `3.691294` | `3.243361` | `1.357` |
| `1a08` | EGNN + typed edges | `42` | linear | `0.153967` | `1.182704` | `3.810299` | `3.060091` | `1.900` |
| `1a08` | EGNN baseline | `42` | linear | `0.117170` | `1.321208` | `4.041913` | `3.140042` | `1.210` |
| `1a08` | heterogeneous frame-based backbone | `42` | linear | `0.095705` | `0.154800` | `2.303952` | `2.175070` | `2.802` |
| `1a09` | EGNN + complete frames | `42` | linear | `0.074038` | `0.074038` | `4.284374` | `2.968930` | `1.406` |
| `1a09` | EGNN + ligand context | `42` | linear | `0.054204` | `0.227505` | `3.545995` | `3.154081` | `1.389` |
| `1a09` | EGNN + typed edges | `42` | linear | `0.115216` | `0.205515` | `2.881196` | `2.767656` | `1.900` |
| `1a09` | EGNN baseline | `42` | linear | `0.082954` | `0.209532` | `3.991216` | `2.681486` | `1.253` |
| `1a09` | heterogeneous frame-based backbone | `42` | linear | `0.091035` | `0.157578` | `2.309657` | `2.281794` | `3.120` |
| `1a0q` | EGNN + complete frames | `42` | linear | `0.117726` | `0.125585` | `2.985805` | `2.120402` | `1.956` |
| `1a0q` | EGNN + ligand context | `42` | linear | `0.111888` | `0.429145` | `2.968071` | `1.873122` | `1.689` |
| `1a0q` | EGNN + typed edges | `42` | linear | `0.109097` | `0.123580` | `3.159559` | `2.017040` | `2.659` |
| `1a0q` | EGNN baseline | `42` | linear | `0.098156` | `0.125443` | `2.813662` | `2.201301` | `1.600` |
| `1a0q` | heterogeneous frame-based backbone | `42` | linear | `0.030627` | `0.030627` | `1.230946` | `1.177538` | `3.894` |
| `1a0t` | EGNN + complete frames | `42` | linear | `0.087331` | `0.313361` | `2.639515` | `1.707345` | `1.509` |
| `1a0t` | EGNN + ligand context | `42` | linear | `0.093365` | `0.131933` | `2.613037` | `2.104104` | `1.604` |
| `1a0t` | EGNN + typed edges | `42` | linear | `0.115563` | `0.427993` | `2.232615` | `1.782276` | `2.082` |
| `1a0t` | EGNN baseline | `42` | linear | `0.108209` | `0.440705` | `2.722568` | `1.850040` | `1.306` |
| `1a0t` | heterogeneous frame-based backbone | `42` | linear | `0.032724` | `0.085755` | `1.220531` | `1.098379` | `3.358` |
| `1a1b` | EGNN + complete frames | `42` | linear | `0.116236` | `0.815986` | `2.739613` | `2.283028` | `1.441` |
| `1a1b` | EGNN + ligand context | `42` | linear | `0.088146` | `0.104681` | `3.838747` | `3.062018` | `1.482` |
| `1a1b` | EGNN + typed edges | `42` | linear | `0.151050` | `0.533669` | `3.703964` | `3.046341` | `2.195` |
| `1a1b` | EGNN baseline | `42` | linear | `0.120310` | `0.126128` | `3.586872` | `3.339651` | `1.250` |
| `1a1b` | heterogeneous frame-based backbone | `42` | linear | `0.085780` | `0.637076` | `2.464023` | `2.398202` | `2.725` |
| `1a1c` | EGNN + complete frames | `42` | linear | `0.120136` | `0.391225` | `3.612282` | `3.108616` | `1.420` |
| `1a1c` | EGNN + ligand context | `42` | linear | `0.141073` | `0.326907` | `3.299597` | `3.025429` | `1.297` |
| `1a1c` | EGNN + typed edges | `42` | linear | `0.115690` | `0.456656` | `3.759594` | `3.082307` | `1.874` |
| `1a1c` | EGNN baseline | `42` | linear | `0.106062` | `0.407836` | `3.911430` | `2.615839` | `1.194` |
| `1a1c` | heterogeneous frame-based backbone | `42` | linear | `0.094399` | `0.205057` | `2.295322` | `2.178056` | `2.811` |
| `1a1e` | EGNN + complete frames | `42` | linear | `0.082487` | `0.474784` | `4.159632` | `2.731913` | `1.466` |
| `1a1e` | EGNN + ligand context | `42` | linear | `0.064845` | `0.137578` | `3.646760` | `3.092303` | `1.388` |
| `1a1e` | EGNN + typed edges | `42` | linear | `0.147514` | `0.848695` | `3.354872` | `2.555600` | `1.833` |
| `1a1e` | EGNN baseline | `42` | linear | `0.116605` | `2.294093` | `3.906642` | `3.360886` | `1.252` |
| `1a1e` | heterogeneous frame-based backbone | `42` | linear | `0.068316` | `0.174944` | `2.225634` | `2.163719` | `2.879` |
| `1a28` | EGNN + complete frames | `42` | linear | `0.076092` | `0.080807` | `2.757669` | `1.617454` | `1.923` |
| `1a28` | EGNN + ligand context | `42` | linear | `0.113290` | `1.330422` | `2.303921` | `1.965732` | `1.625` |
| `1a28` | EGNN + typed edges | `42` | linear | `0.113348` | `1.095797` | `2.522824` | `2.304009` | `2.472` |
| `1a28` | EGNN baseline | `42` | linear | `0.068233` | `0.122401` | `2.706488` | `1.837984` | `1.468` |
| `1a28` | heterogeneous frame-based backbone | `42` | linear | `0.024514` | `0.033103` | `1.099109` | `0.981854` | `4.176` |
| `1a2c` | EGNN + complete frames | `42` | linear | `0.111054` | `0.116244` | `2.725049` | `2.262186` | `1.408` |
| `1a2c` | EGNN + ligand context | `42` | linear | `0.045792` | `0.146940` | `3.315064` | `2.872304` | `1.487` |
| `1a2c` | EGNN + typed edges | `42` | linear | `0.120485` | `0.202623` | `3.250743` | `2.497247` | `2.153` |
| `1a2c` | EGNN baseline | `42` | linear | `0.106584` | `0.197001` | `3.133168` | `2.622637` | `1.288` |
| `1a2c` | heterogeneous frame-based backbone | `42` | linear | `0.056380` | `0.150577` | `1.061612` | `1.026921` | `3.334` |
| `1a30` | EGNN + complete frames | `42` | linear | `0.092305` | `0.495927` | `2.588173` | `1.732424` | `1.673` |
| `1a30` | EGNN + ligand context | `42` | linear | `0.103421` | `1.391530` | `2.854106` | `2.319296` | `1.525` |
| `1a30` | EGNN + typed edges | `42` | linear | `0.110984` | `0.855894` | `2.509354` | `1.910009` | `2.138` |
| `1a30` | EGNN baseline | `42` | linear | `0.097503` | `0.285520` | `2.499769` | `1.874313` | `1.375` |
| `1a30` | heterogeneous frame-based backbone | `42` | linear | `0.014618` | `0.033522` | `1.083982` | `0.986005` | `3.330` |

## Aggregate Means

| Model | Schedule | Complexes | Runs | Mean Raw Ligand RMSE | Std Raw Ligand RMSE | Mean Aligned Ligand RMSD | Std Aligned Ligand RMSD | Success@2A | Success@5A | Mean Training Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EGNN + complete frames | - | `20` | `20` | `2.843765` | `0.770935` | `2.058147` | `0.708740` | `40.0%` | `100.0%` | `1.676` |
| EGNN + ligand context | - | `20` | `20` | `2.900453` | `0.860999` | `2.395254` | `0.863686` | `35.0%` | `100.0%` | `1.552` |
| EGNN + typed edges | - | `20` | `20` | `2.801815` | `0.787542` | `2.260544` | `0.791312` | `35.0%` | `100.0%` | `2.194` |
| EGNN baseline | - | `20` | `20` | `3.164165` | `0.854992` | `2.350469` | `0.868745` | `40.0%` | `100.0%` | `1.409` |
| heterogeneous frame-based backbone | - | `20` | `20` | `1.603221` | `0.518973` | `1.470759` | `0.563780` | `70.0%` | `100.0%` | `3.398` |

## Interpretation

- Across the selected comparison runs, the heterogeneous frame-based backbone reduces mean raw ligand RMSE by `49.3%` relative to the EGNN baseline.
- Across the same runs, the heterogeneous frame-based backbone reduces mean aligned ligand RMSD by `37.4%`.
- Success@2A changes from `40.0%` to `70.0%`.
- The improvement is not free: the heterogeneous frame-based backbone is roughly `2.41x` slower in mean training time.
- This is still a small-sample comparison, so it strengthens the project narrative but does not justify benchmark-scale claims.

## Source Logs

- `10gs` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/10gs_complete_frames_seed42_log.md`
- `10gs` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/10gs_ligand_context_seed42_log.md`
- `10gs` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/10gs_typed_edges_seed42_log.md`
- `10gs` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/10gs_baseline_seed42_log.md`
- `10gs` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/10gs_frame_backbone_seed42_log.md`
- `11gs` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/11gs_complete_frames_seed42_log.md`
- `11gs` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/11gs_ligand_context_seed42_log.md`
- `11gs` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/11gs_typed_edges_seed42_log.md`
- `11gs` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/11gs_baseline_seed42_log.md`
- `11gs` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/11gs_frame_backbone_seed42_log.md`
- `13gs` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/13gs_complete_frames_seed42_log.md`
- `13gs` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/13gs_ligand_context_seed42_log.md`
- `13gs` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/13gs_typed_edges_seed42_log.md`
- `13gs` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/13gs_baseline_seed42_log.md`
- `13gs` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/13gs_frame_backbone_seed42_log.md`
- `16pk` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/16pk_complete_frames_seed42_log.md`
- `16pk` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/16pk_ligand_context_seed42_log.md`
- `16pk` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/16pk_typed_edges_seed42_log.md`
- `16pk` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/16pk_baseline_seed42_log.md`
- `16pk` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/16pk_frame_backbone_seed42_log.md`
- `184l` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/184l_complete_frames_seed42_log.md`
- `184l` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/184l_ligand_context_seed42_log.md`
- `184l` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/184l_typed_edges_seed42_log.md`
- `184l` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/184l_baseline_seed42_log.md`
- `184l` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/184l_frame_backbone_seed42_log.md`
- `185l` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/185l_complete_frames_seed42_log.md`
- `185l` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/185l_ligand_context_seed42_log.md`
- `185l` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/185l_typed_edges_seed42_log.md`
- `185l` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/185l_baseline_seed42_log.md`
- `185l` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/185l_frame_backbone_seed42_log.md`
- `186l` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/186l_complete_frames_seed42_log.md`
- `186l` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/186l_ligand_context_seed42_log.md`
- `186l` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/186l_typed_edges_seed42_log.md`
- `186l` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/186l_baseline_seed42_log.md`
- `186l` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/186l_frame_backbone_seed42_log.md`
- `187l` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/187l_complete_frames_seed42_log.md`
- `187l` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/187l_ligand_context_seed42_log.md`
- `187l` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/187l_typed_edges_seed42_log.md`
- `187l` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/187l_baseline_seed42_log.md`
- `187l` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/187l_frame_backbone_seed42_log.md`
- `188l` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/188l_complete_frames_seed42_log.md`
- `188l` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/188l_ligand_context_seed42_log.md`
- `188l` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/188l_typed_edges_seed42_log.md`
- `188l` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/188l_baseline_seed42_log.md`
- `188l` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/188l_frame_backbone_seed42_log.md`
- `1a07` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/1a07_complete_frames_seed42_log.md`
- `1a07` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/1a07_ligand_context_seed42_log.md`
- `1a07` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/1a07_typed_edges_seed42_log.md`
- `1a07` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a07_baseline_seed42_log.md`
- `1a07` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a07_frame_backbone_seed42_log.md`
- `1a08` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/1a08_complete_frames_seed42_log.md`
- `1a08` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/1a08_ligand_context_seed42_log.md`
- `1a08` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/1a08_typed_edges_seed42_log.md`
- `1a08` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a08_baseline_seed42_log.md`
- `1a08` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a08_frame_backbone_seed42_log.md`
- `1a09` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/1a09_complete_frames_seed42_log.md`
- `1a09` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/1a09_ligand_context_seed42_log.md`
- `1a09` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/1a09_typed_edges_seed42_log.md`
- `1a09` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a09_baseline_seed42_log.md`
- `1a09` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a09_frame_backbone_seed42_log.md`
- `1a0q` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/1a0q_complete_frames_seed42_log.md`
- `1a0q` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/1a0q_ligand_context_seed42_log.md`
- `1a0q` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/1a0q_typed_edges_seed42_log.md`
- `1a0q` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a0q_baseline_seed42_log.md`
- `1a0q` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a0q_frame_backbone_seed42_log.md`
- `1a0t` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/1a0t_complete_frames_seed42_log.md`
- `1a0t` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/1a0t_ligand_context_seed42_log.md`
- `1a0t` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/1a0t_typed_edges_seed42_log.md`
- `1a0t` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a0t_baseline_seed42_log.md`
- `1a0t` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a0t_frame_backbone_seed42_log.md`
- `1a1b` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/1a1b_complete_frames_seed42_log.md`
- `1a1b` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/1a1b_ligand_context_seed42_log.md`
- `1a1b` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/1a1b_typed_edges_seed42_log.md`
- `1a1b` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a1b_baseline_seed42_log.md`
- `1a1b` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a1b_frame_backbone_seed42_log.md`
- `1a1c` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/1a1c_complete_frames_seed42_log.md`
- `1a1c` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/1a1c_ligand_context_seed42_log.md`
- `1a1c` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/1a1c_typed_edges_seed42_log.md`
- `1a1c` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a1c_baseline_seed42_log.md`
- `1a1c` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a1c_frame_backbone_seed42_log.md`
- `1a1e` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/1a1e_complete_frames_seed42_log.md`
- `1a1e` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/1a1e_ligand_context_seed42_log.md`
- `1a1e` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/1a1e_typed_edges_seed42_log.md`
- `1a1e` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a1e_baseline_seed42_log.md`
- `1a1e` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a1e_frame_backbone_seed42_log.md`
- `1a28` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/1a28_complete_frames_seed42_log.md`
- `1a28` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/1a28_ligand_context_seed42_log.md`
- `1a28` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/1a28_typed_edges_seed42_log.md`
- `1a28` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a28_baseline_seed42_log.md`
- `1a28` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a28_frame_backbone_seed42_log.md`
- `1a2c` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/1a2c_complete_frames_seed42_log.md`
- `1a2c` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/1a2c_ligand_context_seed42_log.md`
- `1a2c` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/1a2c_typed_edges_seed42_log.md`
- `1a2c` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a2c_baseline_seed42_log.md`
- `1a2c` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a2c_frame_backbone_seed42_log.md`
- `1a30` EGNN + complete frames seed `42` (linear): `docs/training/panel20/architecture/1a30_complete_frames_seed42_log.md`
- `1a30` EGNN + ligand context seed `42` (linear): `docs/training/panel20/architecture/1a30_ligand_context_seed42_log.md`
- `1a30` EGNN + typed edges seed `42` (linear): `docs/training/panel20/architecture/1a30_typed_edges_seed42_log.md`
- `1a30` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a30_baseline_seed42_log.md`
- `1a30` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a30_frame_backbone_seed42_log.md`
