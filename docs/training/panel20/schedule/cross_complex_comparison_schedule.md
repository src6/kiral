# Cross-Complex Comparison

This summary aggregates the selected real-pair CPU runs already stored in `docs/training/`.
Canonical mode is deterministic once the complex manifest, model set, and seed set are fixed.

## Per-Complex Results

| Complex | Model | Seed | Schedule | Best Loss | Final Loss | Raw Ligand RMSE | Aligned Ligand RMSD | Training Seconds |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `10gs` | EGNN baseline | `42` | cosine | `0.081932` | `1.973334` | `3.128701` | `2.857104` | `1.422` |
| `10gs` | EGNN baseline | `42` | linear | `0.096764` | `0.691036` | `3.616529` | `2.648997` | `1.350` |
| `10gs` | EGNN baseline | `43` | cosine | `0.142441` | `0.229119` | `3.105919` | `2.522544` | `1.366` |
| `10gs` | EGNN baseline | `43` | linear | `0.110406` | `0.347098` | `3.179696` | `2.850913` | `1.349` |
| `10gs` | heterogeneous frame-based backbone | `42` | cosine | `0.037627` | `0.074738` | `1.248284` | `1.053409` | `3.244` |
| `10gs` | heterogeneous frame-based backbone | `42` | linear | `0.039260` | `0.061780` | `1.637889` | `1.420225` | `3.223` |
| `10gs` | heterogeneous frame-based backbone | `43` | cosine | `0.042642` | `0.110958` | `1.303830` | `1.136256` | `3.223` |
| `10gs` | heterogeneous frame-based backbone | `43` | linear | `0.043682` | `0.122036` | `1.633087` | `1.529803` | `3.260` |
| `11gs` | EGNN baseline | `42` | cosine | `0.049167` | `0.052655` | `8.777290` | `8.043284` | `1.398` |
| `11gs` | EGNN baseline | `42` | linear | `0.039134` | `0.045522` | `3.213288` | `2.960820` | `1.271` |
| `11gs` | EGNN baseline | `43` | cosine | `0.168112` | `7.212535` | `3.503502` | `2.972747` | `1.393` |
| `11gs` | EGNN baseline | `43` | linear | `0.119741` | `0.944856` | `3.385298` | `3.096727` | `1.348` |
| `11gs` | heterogeneous frame-based backbone | `42` | cosine | `0.092932` | `1.768753` | `1.542629` | `1.448926` | `3.322` |
| `11gs` | heterogeneous frame-based backbone | `42` | linear | `0.061123` | `0.682064` | `1.849536` | `1.738707` | `3.408` |
| `11gs` | heterogeneous frame-based backbone | `43` | cosine | `0.055421` | `0.110313` | `2.053218` | `1.900694` | `3.799` |
| `11gs` | heterogeneous frame-based backbone | `43` | linear | `0.053327` | `0.113465` | `2.473151` | `2.374116` | `3.665` |
| `13gs` | EGNN baseline | `42` | cosine | `0.068299` | `0.102960` | `5.474442` | `4.281826` | `1.313` |
| `13gs` | EGNN baseline | `42` | linear | `0.080822` | `0.107243` | `4.452593` | `3.240630` | `1.288` |
| `13gs` | EGNN baseline | `43` | cosine | `0.090591` | `7.722885` | `4.073788` | `2.845092` | `1.329` |
| `13gs` | EGNN baseline | `43` | linear | `0.142954` | `1.407864` | `3.598622` | `2.794786` | `1.365` |
| `13gs` | heterogeneous frame-based backbone | `42` | cosine | `0.056775` | `0.086587` | `1.170107` | `1.145876` | `2.955` |
| `13gs` | heterogeneous frame-based backbone | `42` | linear | `0.053906` | `0.055329` | `1.533526` | `1.513411` | `2.859` |
| `13gs` | heterogeneous frame-based backbone | `43` | cosine | `0.050311` | `0.087240` | `1.200804` | `1.021866` | `3.060` |
| `13gs` | heterogeneous frame-based backbone | `43` | linear | `0.059242` | `0.075583` | `1.558498` | `1.448970` | `2.971` |
| `16pk` | EGNN baseline | `42` | cosine | `0.103277` | `1.498551` | `4.902953` | `2.711189` | `1.366` |
| `16pk` | EGNN baseline | `42` | linear | `0.115222` | `0.648591` | `4.935036` | `4.076063` | `1.299` |
| `16pk` | EGNN baseline | `43` | cosine | `0.094004` | `5516.803711` | `3.734845` | `3.343020` | `1.373` |
| `16pk` | EGNN baseline | `43` | linear | `0.107620` | `1.157698` | `3.809259` | `3.561267` | `1.362` |
| `16pk` | heterogeneous frame-based backbone | `42` | cosine | `0.129207` | `2.271089` | `1.582474` | `1.483916` | `3.320` |
| `16pk` | heterogeneous frame-based backbone | `42` | linear | `0.095388` | `0.703123` | `2.310936` | `2.240291` | `3.213` |
| `16pk` | heterogeneous frame-based backbone | `43` | cosine | `0.066539` | `0.179102` | `1.525432` | `1.422971` | `3.650` |
| `16pk` | heterogeneous frame-based backbone | `43` | linear | `0.062540` | `0.169125` | `2.267189` | `2.202529` | `3.289` |
| `184l` | EGNN baseline | `42` | cosine | `0.095666` | `13.224441` | `5.671480` | `4.798329` | `1.797` |
| `184l` | EGNN baseline | `42` | linear | `0.094761` | `1.262720` | `1.871435` | `1.157649` | `1.652` |
| `184l` | EGNN baseline | `43` | cosine | `0.119781` | `0.230276` | `3.938549` | `3.154422` | `3.089` |
| `184l` | EGNN baseline | `43` | linear | `0.096535` | `0.184737` | `2.322452` | `1.485664` | `1.955` |
| `184l` | heterogeneous frame-based backbone | `42` | cosine | `0.013715` | `0.016154` | `1.419393` | `1.189085` | `4.227` |
| `184l` | heterogeneous frame-based backbone | `42` | linear | `0.013456` | `0.029134` | `1.299084` | `1.147352` | `4.382` |
| `184l` | heterogeneous frame-based backbone | `43` | cosine | `0.014003` | `0.014003` | `1.036818` | `0.648680` | `3.981` |
| `184l` | heterogeneous frame-based backbone | `43` | linear | `0.005364` | `0.014175` | `1.008611` | `0.677883` | `3.938` |
| `185l` | EGNN baseline | `42` | cosine | `0.073140` | `4.974591` | `1.595300` | `0.870781` | `1.817` |
| `185l` | EGNN baseline | `42` | linear | `0.069083` | `0.470628` | `2.536903` | `1.480046` | `1.733` |
| `185l` | EGNN baseline | `43` | cosine | `0.115585` | `1.870467` | `2.738154` | `1.486036` | `1.858` |
| `185l` | EGNN baseline | `43` | linear | `0.079286` | `0.284551` | `2.616352` | `1.652825` | `1.829` |
| `185l` | heterogeneous frame-based backbone | `42` | cosine | `0.018156` | `0.052975` | `1.287329` | `1.047950` | `4.059` |
| `185l` | heterogeneous frame-based backbone | `42` | linear | `0.011879` | `0.038537` | `1.204362` | `1.085284` | `3.960` |
| `185l` | heterogeneous frame-based backbone | `43` | cosine | `0.003199` | `0.003813` | `1.160562` | `0.877392` | `4.004` |
| `185l` | heterogeneous frame-based backbone | `43` | linear | `0.003904` | `0.004366` | `0.961796` | `0.737320` | `4.080` |
| `186l` | EGNN baseline | `42` | cosine | `0.092048` | `12.141287` | `2.878291` | `1.194786` | `1.809` |
| `186l` | EGNN baseline | `42` | linear | `0.076203` | `0.943669` | `2.136646` | `1.252245` | `1.727` |
| `186l` | EGNN baseline | `43` | cosine | `0.106459` | `0.138533` | `1.912814` | `1.037800` | `1.805` |
| `186l` | EGNN baseline | `43` | linear | `0.091308` | `0.123328` | `1.879630` | `1.225839` | `1.804` |
| `186l` | heterogeneous frame-based backbone | `42` | cosine | `0.015256` | `0.017940` | `1.002711` | `0.763640` | `4.012` |
| `186l` | heterogeneous frame-based backbone | `42` | linear | `0.005295` | `0.005295` | `0.980356` | `0.801763` | `3.942` |
| `186l` | heterogeneous frame-based backbone | `43` | cosine | `0.012081` | `0.016756` | `0.971788` | `0.744582` | `3.997` |
| `186l` | heterogeneous frame-based backbone | `43` | linear | `0.015062` | `0.028853` | `0.976168` | `0.869672` | `4.022` |
| `187l` | EGNN baseline | `42` | cosine | `0.089165` | `4.229994` | `1.665079` | `0.833012` | `1.836` |
| `187l` | EGNN baseline | `42` | linear | `0.078419` | `0.502475` | `1.842474` | `0.736532` | `1.799` |
| `187l` | EGNN baseline | `43` | cosine | `0.087184` | `1.850784` | `1.792373` | `1.270795` | `1.811` |
| `187l` | EGNN baseline | `43` | linear | `0.083630` | `0.272403` | `1.683415` | `1.209819` | `1.809` |
| `187l` | heterogeneous frame-based backbone | `42` | cosine | `0.013680` | `0.197531` | `1.012589` | `0.851365` | `3.964` |
| `187l` | heterogeneous frame-based backbone | `42` | linear | `0.013376` | `0.167010` | `0.997365` | `0.844008` | `3.949` |
| `187l` | heterogeneous frame-based backbone | `43` | cosine | `0.007298` | `0.081123` | `1.729429` | `0.898785` | `4.086` |
| `187l` | heterogeneous frame-based backbone | `43` | linear | `0.011166` | `0.053824` | `1.542490` | `0.796607` | `3.999` |
| `188l` | EGNN baseline | `42` | cosine | `0.110856` | `0.111916` | `2.543933` | `1.546489` | `1.764` |
| `188l` | EGNN baseline | `42` | linear | `0.070426` | `0.095945` | `2.147055` | `1.281683` | `1.715` |
| `188l` | EGNN baseline | `43` | cosine | `0.075908` | `0.169717` | `2.422995` | `1.463463` | `1.734` |
| `188l` | EGNN baseline | `43` | linear | `0.086047` | `0.118234` | `1.904131` | `1.007534` | `1.761` |
| `188l` | heterogeneous frame-based backbone | `42` | cosine | `0.013516` | `0.202177` | `1.581360` | `0.662302` | `3.892` |
| `188l` | heterogeneous frame-based backbone | `42` | linear | `0.012525` | `0.173573` | `1.276215` | `0.632821` | `3.757` |
| `188l` | heterogeneous frame-based backbone | `43` | cosine | `0.006285` | `0.091901` | `1.455232` | `0.702677` | `3.859` |
| `188l` | heterogeneous frame-based backbone | `43` | linear | `0.008730` | `0.059691` | `1.378383` | `0.738209` | `4.012` |
| `1a07` | EGNN baseline | `42` | cosine | `0.077383` | `1.106414` | `4.398460` | `3.719165` | `1.118` |
| `1a07` | EGNN baseline | `42` | linear | `0.098045` | `0.579624` | `3.217619` | `2.650544` | `1.156` |
| `1a07` | EGNN baseline | `43` | cosine | `0.125824` | `11.613709` | `3.114720` | `2.512500` | `1.328` |
| `1a07` | EGNN baseline | `43` | linear | `0.099008` | `2.292167` | `4.221503` | `3.520947` | `1.180` |
| `1a07` | heterogeneous frame-based backbone | `42` | cosine | `0.067680` | `0.139543` | `1.435343` | `1.216748` | `2.602` |
| `1a07` | heterogeneous frame-based backbone | `42` | linear | `0.065348` | `0.091607` | `1.680381` | `1.523774` | `2.839` |
| `1a07` | heterogeneous frame-based backbone | `43` | cosine | `0.069448` | `0.712099` | `1.734532` | `1.641242` | `2.578` |
| `1a07` | heterogeneous frame-based backbone | `43` | linear | `0.062302` | `0.234865` | `1.837902` | `1.754240` | `2.581` |
| `1a08` | EGNN baseline | `42` | cosine | `0.117900` | `5.945722` | `4.001108` | `2.993229` | `1.223` |
| `1a08` | EGNN baseline | `42` | linear | `0.117170` | `1.321208` | `4.041913` | `3.140042` | `1.210` |
| `1a08` | EGNN baseline | `43` | cosine | `0.072960` | `10.083229` | `2.890070` | `2.506055` | `1.190` |
| `1a08` | EGNN baseline | `43` | linear | `0.091544` | `1.842804` | `2.894538` | `2.619907` | `1.198` |
| `1a08` | heterogeneous frame-based backbone | `42` | cosine | `0.120597` | `0.158904` | `1.711056` | `1.588936` | `2.773` |
| `1a08` | heterogeneous frame-based backbone | `42` | linear | `0.095705` | `0.154800` | `2.303952` | `2.175070` | `2.802` |
| `1a08` | heterogeneous frame-based backbone | `43` | cosine | `0.134068` | `1.108478` | `1.555520` | `1.494122` | `2.814` |
| `1a08` | heterogeneous frame-based backbone | `43` | linear | `0.100942` | `0.357783` | `1.767731` | `1.720452` | `2.783` |
| `1a09` | EGNN baseline | `42` | cosine | `0.136034` | `0.842613` | `4.413311` | `2.358556` | `1.272` |
| `1a09` | EGNN baseline | `42` | linear | `0.082954` | `0.209532` | `3.991216` | `2.681486` | `1.253` |
| `1a09` | EGNN baseline | `43` | cosine | `0.137640` | `12.894392` | `3.320703` | `3.181548` | `1.453` |
| `1a09` | EGNN baseline | `43` | linear | `0.144847` | `2.421634` | `3.320407` | `3.237442` | `1.278` |
| `1a09` | heterogeneous frame-based backbone | `42` | cosine | `0.136581` | `0.145105` | `1.732687` | `1.690613` | `3.074` |
| `1a09` | heterogeneous frame-based backbone | `42` | linear | `0.091035` | `0.157578` | `2.309657` | `2.281794` | `3.120` |
| `1a09` | heterogeneous frame-based backbone | `43` | cosine | `0.121711` | `1.194155` | `1.495880` | `1.442904` | `3.114` |
| `1a09` | heterogeneous frame-based backbone | `43` | linear | `0.101961` | `0.448653` | `1.918061` | `1.877736` | `3.079` |
| `1a0q` | EGNN baseline | `42` | cosine | `0.068387` | `0.068387` | `3.480631` | `2.761304` | `1.517` |
| `1a0q` | EGNN baseline | `42` | linear | `0.098156` | `0.125443` | `2.813662` | `2.201301` | `1.600` |
| `1a0q` | EGNN baseline | `43` | cosine | `0.098433` | `11.471113` | `3.147548` | `2.954657` | `1.475` |
| `1a0q` | EGNN baseline | `43` | linear | `0.102323` | `1.020002` | `2.692614` | `2.524938` | `1.501` |
| `1a0q` | heterogeneous frame-based backbone | `42` | cosine | `0.035944` | `0.055562` | `1.179669` | `1.072221` | `3.582` |
| `1a0q` | heterogeneous frame-based backbone | `42` | linear | `0.030627` | `0.030627` | `1.230946` | `1.177538` | `3.894` |
| `1a0q` | heterogeneous frame-based backbone | `43` | cosine | `0.027352` | `0.049610` | `1.146870` | `0.922384` | `3.692` |
| `1a0q` | heterogeneous frame-based backbone | `43` | linear | `0.016232` | `0.037205` | `1.346104` | `1.198727` | `3.561` |
| `1a0t` | EGNN baseline | `42` | cosine | `0.152814` | `2.045854` | `5.223437` | `3.857907` | `1.325` |
| `1a0t` | EGNN baseline | `42` | linear | `0.108209` | `0.440705` | `2.722568` | `1.850040` | `1.306` |
| `1a0t` | EGNN baseline | `43` | cosine | `0.150930` | `11.176135` | `2.904530` | `1.504512` | `1.307` |
| `1a0t` | EGNN baseline | `43` | linear | `0.127081` | `0.912846` | `2.439105` | `1.797226` | `1.316` |
| `1a0t` | heterogeneous frame-based backbone | `42` | cosine | `0.041467` | `0.086069` | `1.315417` | `1.124498` | `2.981` |
| `1a0t` | heterogeneous frame-based backbone | `42` | linear | `0.032724` | `0.085755` | `1.220531` | `1.098379` | `3.358` |
| `1a0t` | heterogeneous frame-based backbone | `43` | cosine | `0.019509` | `0.078618` | `1.200197` | `1.151417` | `2.966` |
| `1a0t` | heterogeneous frame-based backbone | `43` | linear | `0.024668` | `0.077089` | `1.225224` | `1.168527` | `2.944` |
| `1a1b` | EGNN baseline | `42` | cosine | `0.121498` | `0.121498` | `3.952600` | `3.657962` | `1.150` |
| `1a1b` | EGNN baseline | `42` | linear | `0.120310` | `0.126128` | `3.586872` | `3.339651` | `1.250` |
| `1a1b` | EGNN baseline | `43` | cosine | `0.111441` | `0.113843` | `4.881916` | `2.809714` | `1.181` |
| `1a1b` | EGNN baseline | `43` | linear | `0.084477` | `0.127024` | `4.169044` | `2.907061` | `1.187` |
| `1a1b` | heterogeneous frame-based backbone | `42` | cosine | `0.118257` | `1.973182` | `1.917848` | `1.835999` | `2.706` |
| `1a1b` | heterogeneous frame-based backbone | `42` | linear | `0.085780` | `0.637076` | `2.464023` | `2.398202` | `2.725` |
| `1a1b` | heterogeneous frame-based backbone | `43` | cosine | `0.051488` | `0.127504` | `2.075653` | `1.747348` | `2.730` |
| `1a1b` | heterogeneous frame-based backbone | `43` | linear | `0.052872` | `0.110354` | `2.542195` | `2.279844` | `2.743` |
| `1a1c` | EGNN baseline | `42` | cosine | `0.120261` | `0.251803` | `4.142863` | `2.580978` | `1.182` |
| `1a1c` | EGNN baseline | `42` | linear | `0.106062` | `0.407836` | `3.911430` | `2.615839` | `1.194` |
| `1a1c` | EGNN baseline | `43` | cosine | `0.111413` | `0.139065` | `4.069446` | `3.221747` | `1.252` |
| `1a1c` | EGNN baseline | `43` | linear | `0.135525` | `0.246642` | `4.923056` | `3.431293` | `1.186` |
| `1a1c` | heterogeneous frame-based backbone | `42` | cosine | `0.144846` | `0.330594` | `1.611588` | `1.461245` | `2.783` |
| `1a1c` | heterogeneous frame-based backbone | `42` | linear | `0.094399` | `0.205057` | `2.295322` | `2.178056` | `2.811` |
| `1a1c` | heterogeneous frame-based backbone | `43` | cosine | `0.153759` | `2.004492` | `1.434043` | `1.350489` | `2.840` |
| `1a1c` | heterogeneous frame-based backbone | `43` | linear | `0.122836` | `0.589676` | `2.144232` | `2.048654` | `2.872` |
| `1a1e` | EGNN baseline | `42` | cosine | `0.117287` | `12.626784` | `4.431666` | `3.725410` | `1.077` |
| `1a1e` | EGNN baseline | `42` | linear | `0.116605` | `2.294093` | `3.906642` | `3.360886` | `1.252` |
| `1a1e` | EGNN baseline | `43` | cosine | `0.136908` | `0.155574` | `3.635628` | `3.190251` | `1.133` |
| `1a1e` | EGNN baseline | `43` | linear | `0.133432` | `0.151429` | `6.609172` | `6.021303` | `1.177` |
| `1a1e` | heterogeneous frame-based backbone | `42` | cosine | `0.112651` | `0.360279` | `1.558364` | `1.485989` | `2.486` |
| `1a1e` | heterogeneous frame-based backbone | `42` | linear | `0.068316` | `0.174944` | `2.225634` | `2.163719` | `2.879` |
| `1a1e` | heterogeneous frame-based backbone | `43` | cosine | `0.120230` | `1.066052` | `1.350033` | `1.305012` | `2.754` |
| `1a1e` | heterogeneous frame-based backbone | `43` | linear | `0.069059` | `0.442092` | `1.975094` | `1.919539` | `2.566` |
| `1a28` | EGNN baseline | `42` | cosine | `0.053843` | `0.139436` | `3.070183` | `1.920651` | `1.324` |
| `1a28` | EGNN baseline | `42` | linear | `0.068233` | `0.122401` | `2.706488` | `1.837984` | `1.468` |
| `1a28` | EGNN baseline | `43` | cosine | `0.088077` | `0.134040` | `3.234096` | `1.654401` | `1.450` |
| `1a28` | EGNN baseline | `43` | linear | `0.087401` | `0.087401` | `3.375755` | `1.677637` | `1.447` |
| `1a28` | heterogeneous frame-based backbone | `42` | cosine | `0.022334` | `0.037046` | `1.107540` | `0.961392` | `3.462` |
| `1a28` | heterogeneous frame-based backbone | `42` | linear | `0.024514` | `0.033103` | `1.099109` | `0.981854` | `4.176` |
| `1a28` | heterogeneous frame-based backbone | `43` | cosine | `0.022940` | `0.046731` | `1.026716` | `0.966706` | `3.221` |
| `1a28` | heterogeneous frame-based backbone | `43` | linear | `0.018093` | `0.038283` | `1.077655` | `1.018271` | `3.355` |
| `1a2c` | EGNN baseline | `42` | cosine | `0.160943` | `0.920163` | `3.205431` | `2.386182` | `1.111` |
| `1a2c` | EGNN baseline | `42` | linear | `0.106584` | `0.197001` | `3.133168` | `2.622637` | `1.288` |
| `1a2c` | EGNN baseline | `43` | cosine | `0.110181` | `10.970328` | `3.215687` | `2.668819` | `1.076` |
| `1a2c` | EGNN baseline | `43` | linear | `0.086038` | `1.833579` | `3.216801` | `2.872355` | `1.114` |
| `1a2c` | heterogeneous frame-based backbone | `42` | cosine | `0.053217` | `0.306954` | `1.111037` | `1.070039` | `2.948` |
| `1a2c` | heterogeneous frame-based backbone | `42` | linear | `0.056380` | `0.150577` | `1.061612` | `1.026921` | `3.334` |
| `1a2c` | heterogeneous frame-based backbone | `43` | cosine | `0.049150` | `0.049150` | `1.518995` | `1.481001` | `2.855` |
| `1a2c` | heterogeneous frame-based backbone | `43` | linear | `0.057504` | `0.063931` | `1.632159` | `1.603623` | `2.935` |
| `1a30` | EGNN baseline | `42` | cosine | `0.162562` | `0.703214` | `2.771368` | `2.015371` | `1.567` |
| `1a30` | EGNN baseline | `42` | linear | `0.097503` | `0.285520` | `2.499769` | `1.874313` | `1.375` |
| `1a30` | EGNN baseline | `43` | cosine | `0.110315` | `1.029338` | `2.410668` | `1.873860` | `1.477` |
| `1a30` | EGNN baseline | `43` | linear | `0.122377` | `0.299881` | `2.618954` | `1.884227` | `1.506` |
| `1a30` | heterogeneous frame-based backbone | `42` | cosine | `0.019455` | `0.026995` | `1.081148` | `0.998237` | `3.937` |
| `1a30` | heterogeneous frame-based backbone | `42` | linear | `0.014618` | `0.033522` | `1.083982` | `0.986005` | `3.330` |
| `1a30` | heterogeneous frame-based backbone | `43` | cosine | `0.028150` | `0.086558` | `1.182076` | `0.949128` | `3.526` |
| `1a30` | heterogeneous frame-based backbone | `43` | linear | `0.027046` | `0.056447` | `1.187476` | `1.000063` | `3.618` |

## Aggregate Means

| Model | Schedule | Complexes | Runs | Mean Raw Ligand RMSE | Std Raw Ligand RMSE | Mean Aligned Ligand RMSD | Std Aligned Ligand RMSD | Success@2A | Success@5A | Mean Training Seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EGNN baseline | cosine | `20` | `40` | `3.594412` | `1.286367` | `2.682187` | `1.267821` | `30.0%` | `97.5%` | `1.462` |
| EGNN baseline | linear | `20` | `40` | `3.203578` | `0.999132` | `2.459727` | `1.014309` | `40.0%` | `97.5%` | `1.421` |
| heterogeneous frame-based backbone | cosine | `20` | `40` | `1.394155` | `0.284653` | `1.198951` | `0.329917` | `100.0%` | `100.0%` | `3.327` |
| heterogeneous frame-based backbone | linear | `20` | `40` | `1.612941` | `0.498714` | `1.459499` | `0.555293` | `75.0%` | `100.0%` | `3.356` |

## Interpretation

- Across the selected comparison runs, the heterogeneous frame-based backbone reduces mean raw ligand RMSE by `49.7%` relative to the EGNN baseline.
- Across the same runs, the heterogeneous frame-based backbone reduces mean aligned ligand RMSD by `40.7%`.
- Success@2A changes from `40.0%` to `75.0%`.
- The improvement is not free: the heterogeneous frame-based backbone is roughly `2.36x` slower in mean training time.
- This is still a small-sample comparison, so it strengthens the project narrative but does not justify benchmark-scale claims.

## Source Logs

- `10gs` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/10gs_baseline_cosine_seed42_log.md`
- `10gs` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/10gs_baseline_seed42_log.md`
- `10gs` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/10gs_baseline_cosine_seed43_log.md`
- `10gs` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/10gs_baseline_linear_seed43_log.md`
- `10gs` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/10gs_frame_backbone_cosine_seed42_log.md`
- `10gs` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/10gs_frame_backbone_seed42_log.md`
- `10gs` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/10gs_frame_backbone_cosine_seed43_log.md`
- `10gs` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/10gs_frame_backbone_linear_seed43_log.md`
- `11gs` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/11gs_baseline_cosine_seed42_log.md`
- `11gs` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/11gs_baseline_seed42_log.md`
- `11gs` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/11gs_baseline_cosine_seed43_log.md`
- `11gs` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/11gs_baseline_linear_seed43_log.md`
- `11gs` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/11gs_frame_backbone_cosine_seed42_log.md`
- `11gs` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/11gs_frame_backbone_seed42_log.md`
- `11gs` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/11gs_frame_backbone_cosine_seed43_log.md`
- `11gs` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/11gs_frame_backbone_linear_seed43_log.md`
- `13gs` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/13gs_baseline_cosine_seed42_log.md`
- `13gs` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/13gs_baseline_seed42_log.md`
- `13gs` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/13gs_baseline_cosine_seed43_log.md`
- `13gs` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/13gs_baseline_linear_seed43_log.md`
- `13gs` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/13gs_frame_backbone_cosine_seed42_log.md`
- `13gs` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/13gs_frame_backbone_seed42_log.md`
- `13gs` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/13gs_frame_backbone_cosine_seed43_log.md`
- `13gs` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/13gs_frame_backbone_linear_seed43_log.md`
- `16pk` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/16pk_baseline_cosine_seed42_log.md`
- `16pk` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/16pk_baseline_seed42_log.md`
- `16pk` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/16pk_baseline_cosine_seed43_log.md`
- `16pk` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/16pk_baseline_linear_seed43_log.md`
- `16pk` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/16pk_frame_backbone_cosine_seed42_log.md`
- `16pk` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/16pk_frame_backbone_seed42_log.md`
- `16pk` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/16pk_frame_backbone_cosine_seed43_log.md`
- `16pk` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/16pk_frame_backbone_linear_seed43_log.md`
- `184l` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/184l_baseline_cosine_seed42_log.md`
- `184l` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/184l_baseline_seed42_log.md`
- `184l` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/184l_baseline_cosine_seed43_log.md`
- `184l` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/184l_baseline_linear_seed43_log.md`
- `184l` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/184l_frame_backbone_cosine_seed42_log.md`
- `184l` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/184l_frame_backbone_seed42_log.md`
- `184l` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/184l_frame_backbone_cosine_seed43_log.md`
- `184l` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/184l_frame_backbone_linear_seed43_log.md`
- `185l` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/185l_baseline_cosine_seed42_log.md`
- `185l` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/185l_baseline_seed42_log.md`
- `185l` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/185l_baseline_cosine_seed43_log.md`
- `185l` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/185l_baseline_linear_seed43_log.md`
- `185l` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/185l_frame_backbone_cosine_seed42_log.md`
- `185l` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/185l_frame_backbone_seed42_log.md`
- `185l` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/185l_frame_backbone_cosine_seed43_log.md`
- `185l` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/185l_frame_backbone_linear_seed43_log.md`
- `186l` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/186l_baseline_cosine_seed42_log.md`
- `186l` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/186l_baseline_seed42_log.md`
- `186l` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/186l_baseline_cosine_seed43_log.md`
- `186l` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/186l_baseline_linear_seed43_log.md`
- `186l` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/186l_frame_backbone_cosine_seed42_log.md`
- `186l` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/186l_frame_backbone_seed42_log.md`
- `186l` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/186l_frame_backbone_cosine_seed43_log.md`
- `186l` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/186l_frame_backbone_linear_seed43_log.md`
- `187l` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/187l_baseline_cosine_seed42_log.md`
- `187l` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/187l_baseline_seed42_log.md`
- `187l` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/187l_baseline_cosine_seed43_log.md`
- `187l` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/187l_baseline_linear_seed43_log.md`
- `187l` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/187l_frame_backbone_cosine_seed42_log.md`
- `187l` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/187l_frame_backbone_seed42_log.md`
- `187l` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/187l_frame_backbone_cosine_seed43_log.md`
- `187l` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/187l_frame_backbone_linear_seed43_log.md`
- `188l` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/188l_baseline_cosine_seed42_log.md`
- `188l` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/188l_baseline_seed42_log.md`
- `188l` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/188l_baseline_cosine_seed43_log.md`
- `188l` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/188l_baseline_linear_seed43_log.md`
- `188l` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/188l_frame_backbone_cosine_seed42_log.md`
- `188l` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/188l_frame_backbone_seed42_log.md`
- `188l` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/188l_frame_backbone_cosine_seed43_log.md`
- `188l` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/188l_frame_backbone_linear_seed43_log.md`
- `1a07` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/1a07_baseline_cosine_seed42_log.md`
- `1a07` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a07_baseline_seed42_log.md`
- `1a07` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/1a07_baseline_cosine_seed43_log.md`
- `1a07` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/1a07_baseline_linear_seed43_log.md`
- `1a07` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/1a07_frame_backbone_cosine_seed42_log.md`
- `1a07` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a07_frame_backbone_seed42_log.md`
- `1a07` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/1a07_frame_backbone_cosine_seed43_log.md`
- `1a07` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/1a07_frame_backbone_linear_seed43_log.md`
- `1a08` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/1a08_baseline_cosine_seed42_log.md`
- `1a08` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a08_baseline_seed42_log.md`
- `1a08` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/1a08_baseline_cosine_seed43_log.md`
- `1a08` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/1a08_baseline_linear_seed43_log.md`
- `1a08` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/1a08_frame_backbone_cosine_seed42_log.md`
- `1a08` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a08_frame_backbone_seed42_log.md`
- `1a08` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/1a08_frame_backbone_cosine_seed43_log.md`
- `1a08` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/1a08_frame_backbone_linear_seed43_log.md`
- `1a09` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/1a09_baseline_cosine_seed42_log.md`
- `1a09` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a09_baseline_seed42_log.md`
- `1a09` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/1a09_baseline_cosine_seed43_log.md`
- `1a09` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/1a09_baseline_linear_seed43_log.md`
- `1a09` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/1a09_frame_backbone_cosine_seed42_log.md`
- `1a09` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a09_frame_backbone_seed42_log.md`
- `1a09` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/1a09_frame_backbone_cosine_seed43_log.md`
- `1a09` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/1a09_frame_backbone_linear_seed43_log.md`
- `1a0q` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/1a0q_baseline_cosine_seed42_log.md`
- `1a0q` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a0q_baseline_seed42_log.md`
- `1a0q` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/1a0q_baseline_cosine_seed43_log.md`
- `1a0q` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/1a0q_baseline_linear_seed43_log.md`
- `1a0q` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/1a0q_frame_backbone_cosine_seed42_log.md`
- `1a0q` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a0q_frame_backbone_seed42_log.md`
- `1a0q` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/1a0q_frame_backbone_cosine_seed43_log.md`
- `1a0q` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/1a0q_frame_backbone_linear_seed43_log.md`
- `1a0t` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/1a0t_baseline_cosine_seed42_log.md`
- `1a0t` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a0t_baseline_seed42_log.md`
- `1a0t` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/1a0t_baseline_cosine_seed43_log.md`
- `1a0t` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/1a0t_baseline_linear_seed43_log.md`
- `1a0t` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/1a0t_frame_backbone_cosine_seed42_log.md`
- `1a0t` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a0t_frame_backbone_seed42_log.md`
- `1a0t` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/1a0t_frame_backbone_cosine_seed43_log.md`
- `1a0t` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/1a0t_frame_backbone_linear_seed43_log.md`
- `1a1b` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/1a1b_baseline_cosine_seed42_log.md`
- `1a1b` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a1b_baseline_seed42_log.md`
- `1a1b` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/1a1b_baseline_cosine_seed43_log.md`
- `1a1b` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/1a1b_baseline_linear_seed43_log.md`
- `1a1b` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/1a1b_frame_backbone_cosine_seed42_log.md`
- `1a1b` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a1b_frame_backbone_seed42_log.md`
- `1a1b` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/1a1b_frame_backbone_cosine_seed43_log.md`
- `1a1b` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/1a1b_frame_backbone_linear_seed43_log.md`
- `1a1c` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/1a1c_baseline_cosine_seed42_log.md`
- `1a1c` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a1c_baseline_seed42_log.md`
- `1a1c` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/1a1c_baseline_cosine_seed43_log.md`
- `1a1c` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/1a1c_baseline_linear_seed43_log.md`
- `1a1c` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/1a1c_frame_backbone_cosine_seed42_log.md`
- `1a1c` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a1c_frame_backbone_seed42_log.md`
- `1a1c` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/1a1c_frame_backbone_cosine_seed43_log.md`
- `1a1c` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/1a1c_frame_backbone_linear_seed43_log.md`
- `1a1e` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/1a1e_baseline_cosine_seed42_log.md`
- `1a1e` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a1e_baseline_seed42_log.md`
- `1a1e` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/1a1e_baseline_cosine_seed43_log.md`
- `1a1e` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/1a1e_baseline_linear_seed43_log.md`
- `1a1e` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/1a1e_frame_backbone_cosine_seed42_log.md`
- `1a1e` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a1e_frame_backbone_seed42_log.md`
- `1a1e` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/1a1e_frame_backbone_cosine_seed43_log.md`
- `1a1e` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/1a1e_frame_backbone_linear_seed43_log.md`
- `1a28` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/1a28_baseline_cosine_seed42_log.md`
- `1a28` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a28_baseline_seed42_log.md`
- `1a28` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/1a28_baseline_cosine_seed43_log.md`
- `1a28` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/1a28_baseline_linear_seed43_log.md`
- `1a28` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/1a28_frame_backbone_cosine_seed42_log.md`
- `1a28` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a28_frame_backbone_seed42_log.md`
- `1a28` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/1a28_frame_backbone_cosine_seed43_log.md`
- `1a28` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/1a28_frame_backbone_linear_seed43_log.md`
- `1a2c` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/1a2c_baseline_cosine_seed42_log.md`
- `1a2c` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a2c_baseline_seed42_log.md`
- `1a2c` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/1a2c_baseline_cosine_seed43_log.md`
- `1a2c` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/1a2c_baseline_linear_seed43_log.md`
- `1a2c` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/1a2c_frame_backbone_cosine_seed42_log.md`
- `1a2c` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a2c_frame_backbone_seed42_log.md`
- `1a2c` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/1a2c_frame_backbone_cosine_seed43_log.md`
- `1a2c` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/1a2c_frame_backbone_linear_seed43_log.md`
- `1a30` EGNN baseline seed `42` (cosine): `docs/training/panel20/schedule/1a30_baseline_cosine_seed42_log.md`
- `1a30` EGNN baseline seed `42` (linear): `docs/training/panel20/architecture/1a30_baseline_seed42_log.md`
- `1a30` EGNN baseline seed `43` (cosine): `docs/training/panel20/schedule/1a30_baseline_cosine_seed43_log.md`
- `1a30` EGNN baseline seed `43` (linear): `docs/training/panel20/schedule/1a30_baseline_linear_seed43_log.md`
- `1a30` heterogeneous frame-based backbone seed `42` (cosine): `docs/training/panel20/schedule/1a30_frame_backbone_cosine_seed42_log.md`
- `1a30` heterogeneous frame-based backbone seed `42` (linear): `docs/training/panel20/architecture/1a30_frame_backbone_seed42_log.md`
- `1a30` heterogeneous frame-based backbone seed `43` (cosine): `docs/training/panel20/schedule/1a30_frame_backbone_cosine_seed43_log.md`
- `1a30` heterogeneous frame-based backbone seed `43` (linear): `docs/training/panel20/schedule/1a30_frame_backbone_linear_seed43_log.md`
