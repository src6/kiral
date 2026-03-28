# Experiment Comparison

- Left logs: `docs/training/panel20/schedule/*_log.md`
- Right logs: `docs/training/panel20/schedule_cpu_rerun/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `40`
- Compared complexes: `20`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `-0.032511` |
| Mean Final Loss Delta | `-` | `-` | `-0.173476` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `-0.021039` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `-0.025318` |
| Mean Training Seconds Delta | `-` | `-` | `-0.171` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10gs` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.276057` | `1.274844` | `-0.001213` | `1.094832` | `1.072724` | `-0.022108` | `3.234` | `3.172` | `-0.061` |
| `11gs` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.797924` | `1.806685` | `+0.008761` | `1.674810` | `1.673741` | `-0.001070` | `3.561` | `3.154` | `-0.407` |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.185455` | `1.152001` | `-0.033454` | `1.083871` | `1.037124` | `-0.046747` | `3.008` | `2.851` | `-0.156` |
| `16pk` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.553953` | `1.557983` | `+0.004030` | `1.453444` | `1.451150` | `-0.002294` | `3.485` | `3.131` | `-0.353` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.228105` | `1.210825` | `-0.017280` | `0.918883` | `0.902003` | `-0.016880` | `4.104` | `3.732` | `-0.371` |
| `185l` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.223946` | `1.226141` | `+0.002196` | `0.962671` | `0.971563` | `+0.008892` | `4.031` | `3.854` | `-0.178` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `2` | `0.987249` | `0.971861` | `-0.015388` | `0.754111` | `0.742636` | `-0.011475` | `4.005` | `3.859` | `-0.146` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.371009` | `1.367373` | `-0.003637` | `0.875075` | `0.861601` | `-0.013474` | `4.025` | `3.832` | `-0.194` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.518296` | `1.543445` | `+0.025149` | `0.682489` | `0.717959` | `+0.035469` | `3.875` | `3.720` | `-0.156` |
| `1a07` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.584938` | `1.492508` | `-0.092430` | `1.428995` | `1.348864` | `-0.080131` | `2.590` | `2.431` | `-0.159` |
| `1a08` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.633288` | `1.573788` | `-0.059500` | `1.541529` | `1.446937` | `-0.094592` | `2.793` | `2.702` | `-0.092` |
| `1a09` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.614283` | `1.468196` | `-0.146087` | `1.566758` | `1.424705` | `-0.142053` | `3.094` | `2.904` | `-0.190` |
| `1a0q` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.163270` | `1.191724` | `+0.028454` | `0.997302` | `1.021838` | `+0.024535` | `3.637` | `3.353` | `-0.284` |
| `1a0t` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.257807` | `1.169677` | `-0.088131` | `1.137957` | `1.049335` | `-0.088622` | `2.974` | `2.817` | `-0.156` |
| `1a1b` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.996751` | `1.904584` | `-0.092167` | `1.791673` | `1.667510` | `-0.124163` | `2.718` | `2.577` | `-0.141` |
| `1a1c` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.522816` | `1.622287` | `+0.099472` | `1.405867` | `1.530055` | `+0.124188` | `2.811` | `2.620` | `-0.192` |
| `1a1e` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.454198` | `1.433183` | `-0.021016` | `1.395501` | `1.362801` | `-0.032700` | `2.620` | `2.657` | `+0.037` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.067128` | `1.078768` | `+0.011640` | `0.964049` | `0.975871` | `+0.011822` | `3.341` | `3.401` | `+0.059` |
| `1a2c` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.315016` | `1.272893` | `-0.042123` | `1.275520` | `1.229755` | `-0.045765` | `2.901` | `3.004` | `+0.103` |
| `1a30` | heterogeneous frame-based backbone | cosine | `all` | `2` | `1.131612` | `1.143562` | `+0.011950` | `0.973683` | `0.984496` | `+0.010813` | `3.731` | `3.341` | `-0.390` |

## Missing Pairs

No missing pairs.
