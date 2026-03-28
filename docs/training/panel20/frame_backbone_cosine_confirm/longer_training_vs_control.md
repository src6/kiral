# Experiment Comparison

- Left logs: `docs/training/panel20/frame_backbone_cosine_confirm/control/*_log.md`
- Right logs: `docs/training/panel20/frame_backbone_cosine_confirm/longer_training/*_log.md`
- Aggregate mode: `complex`
- Compared runs: `60`
- Compared complexes: `20`

## Aggregate Deltas

| Metric | Left | Right | Delta |
| --- | ---: | ---: | ---: |
| Success@2A | `100.0%` | `100.0%` | `+0.0%` |
| Success@5A | `100.0%` | `100.0%` | `+0.0%` |
| Mean Best Loss Delta | `-` | `-` | `-0.013028` |
| Mean Final Loss Delta | `-` | `-` | `-0.096524` |
| Mean Raw Ligand RMSE Delta | `-` | `-` | `-0.042488` |
| Mean Aligned Ligand RMSD Delta | `-` | `-` | `-0.064899` |
| Mean Training Seconds Delta | `-` | `-` | `+3.864` |

## Paired Rows

| Complex | Model | Schedule | Seed | Matched Runs | Left Raw RMSE | Right Raw RMSE | Delta Raw RMSE | Left Aligned RMSD | Right Aligned RMSD | Delta Aligned RMSD | Left Train s | Right Train s | Delta Train s |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `10gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.226267` | `1.129503` | `-0.096764` | `1.074477` | `0.995627` | `-0.078849` | `3.341` | `6.956` | `+3.615` |
| `11gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.732773` | `1.291697` | `-0.441076` | `1.602874` | `1.179013` | `-0.423861` | `3.288` | `7.274` | `+3.985` |
| `13gs` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.184201` | `1.200424` | `+0.016223` | `1.033504` | `1.105163` | `+0.071660` | `2.998` | `6.493` | `+3.495` |
| `16pk` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.500198` | `1.437836` | `-0.062361` | `1.399697` | `1.208211` | `-0.191485` | `3.270` | `7.314` | `+4.044` |
| `184l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.078402` | `1.414339` | `+0.335937` | `0.801418` | `0.990572` | `+0.189154` | `3.944` | `8.243` | `+4.299` |
| `185l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.321995` | `1.541627` | `+0.219632` | `0.990792` | `0.870828` | `-0.119965` | `4.080` | `9.080` | `+5.000` |
| `186l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `0.968042` | `1.575146` | `+0.607104` | `0.784783` | `0.992144` | `+0.207361` | `4.048` | `8.605` | `+4.557` |
| `187l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.454817` | `1.449760` | `-0.005057` | `0.863725` | `1.018061` | `+0.154335` | `3.985` | `8.336` | `+4.352` |
| `188l` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.351718` | `1.091747` | `-0.259971` | `0.703451` | `0.843585` | `+0.140134` | `3.845` | `8.344` | `+4.499` |
| `1a07` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.419310` | `1.244614` | `-0.174696` | `1.287035` | `1.145782` | `-0.141253` | `2.607` | `5.461` | `+2.855` |
| `1a08` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.715530` | `1.492086` | `-0.223443` | `1.615561` | `1.389131` | `-0.226430` | `2.798` | `6.160` | `+3.361` |
| `1a09` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.518060` | `1.489979` | `-0.028081` | `1.451719` | `1.400465` | `-0.051254` | `3.032` | `6.583` | `+3.552` |
| `1a0q` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.153957` | `1.189296` | `+0.035339` | `0.984591` | `0.977101` | `-0.007490` | `3.484` | `7.717` | `+4.232` |
| `1a0t` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.120275` | `1.056036` | `-0.064239` | `1.016311` | `0.928278` | `-0.088033` | `2.929` | `6.601` | `+3.672` |
| `1a1b` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.846018` | `1.506004` | `-0.340014` | `1.618844` | `1.335370` | `-0.283474` | `2.775` | `6.120` | `+3.345` |
| `1a1c` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.468198` | `1.322071` | `-0.146127` | `1.385941` | `1.185720` | `-0.200221` | `2.741` | `6.025` | `+3.283` |
| `1a1e` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.482855` | `1.348956` | `-0.133899` | `1.406295` | `1.215561` | `-0.190734` | `2.790` | `6.032` | `+3.242` |
| `1a28` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.113274` | `1.155772` | `+0.042498` | `0.972127` | `1.025143` | `+0.053016` | `3.585` | `7.880` | `+4.295` |
| `1a2c` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.231268` | `1.231561` | `+0.000292` | `1.194376` | `1.122276` | `-0.072100` | `3.221` | `6.762` | `+3.541` |
| `1a30` | heterogeneous frame-based backbone | cosine | `all` | `3` | `1.193753` | `1.062690` | `-0.131063` | `1.020509` | `0.982027` | `-0.038482` | `3.485` | `7.545` | `+4.060` |

## Missing Pairs

No missing pairs.
