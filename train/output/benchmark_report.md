# Intent Classifier Benchmark Report

Datasets evaluated: diverse, full

## Model sizes

| Model                                           |   Size (MB) | Params   |
|:------------------------------------------------|------------:|:---------|
| model_mono_ca_MrBERT-ca                         |       56.18 | 12.94 M  |
| model_mono_ca_distilroberta-ca                  |       56.24 | 12.97 M  |
| model_mono_ca_roberta-base-ca                   |       56.24 | 12.97 M  |
| model_mono_ca_roberta-large-ca                  |       56.24 | 12.97 M  |
| model_mono_es_MrBERT-biomed                     |      265.78 | 66.01 M  |
| model_mono_es_MrBERT-es                         |       55.78 | 12.92 M  |
| model_mono_es_MrBERT-legal                      |      265.78 | 66.01 M  |
| model_mono_es_MrBERT-science                    |      265.78 | 66.01 M  |
| model_mono_gl_bertinho-gl                       |       35.2  | 7.74 M   |
| model_mono_pt_bert-base-pt                      |       35.47 | 7.66 M   |
| model_mono_pt_bert-large-pt                     |       35.47 | 7.66 M   |
| model_mono_pt_bertha-pt-small                   |       35.83 | 7.74 M   |
| model_mul_LaBSE                                 |      518.49 | 129.27 M |
| model_mul_MrBERT                                |      267.44 | 66.01 M  |
| model_mul_bert-base-multilingual-cased          |      128.16 | 30.82 M  |
| model_mul_multilingual-e5-small                 |      262.05 | 64.50 M  |
| model_mul_paraphrase-multilingual-MiniLM-L12-v2 |      262.05 | 64.50 M  |
| model_mul_potion-multilingual-128M              |      518.49 | 129.09 M |

## Overall metrics

| model                                           | dataset   |   n_samples |   accuracy |   f1_weighted |   f1_macro |   throughput_sps |
|:------------------------------------------------|:----------|------------:|-----------:|--------------:|-----------:|-----------------:|
| model_mul_potion-multilingual-128M              | diverse   |         960 |     0.9385 |        0.9378 |     0.9385 |            12386 |
| model_mul_LaBSE                                 | diverse   |         960 |     0.9375 |        0.9355 |     0.9365 |            23349 |
| model_mul_bert-base-multilingual-cased          | diverse   |         960 |     0.9177 |        0.9174 |     0.9179 |            16979 |
| model_mul_multilingual-e5-small                 | diverse   |         960 |     0.9146 |        0.914  |     0.9147 |            16446 |
| model_mul_paraphrase-multilingual-MiniLM-L12-v2 | diverse   |         960 |     0.9094 |        0.9087 |     0.9093 |             6569 |
| model_mul_MrBERT                                | diverse   |         960 |     0.5719 |        0.5055 |     0.4969 |            14065 |
| model_mono_pt_bert-base-pt                      | diverse   |         590 |     0.2983 |        0.3468 |     0.3468 |            12098 |
| model_mono_ca_roberta-base-ca                   | diverse   |         505 |     0.3149 |        0.3384 |     0.3384 |            13041 |
| model_mono_pt_bertha-pt-small                   | diverse   |         590 |     0.2847 |        0.3341 |     0.3341 |             8872 |
| model_mono_pt_bert-large-pt                     | diverse   |         590 |     0.2847 |        0.3278 |     0.3278 |            12878 |
| model_mono_ca_distilroberta-ca                  | diverse   |         505 |     0.3069 |        0.3238 |     0.3238 |             7963 |
| model_mono_es_MrBERT-es                         | diverse   |         405 |     0.2938 |        0.3219 |     0.3219 |            16627 |
| model_mono_ca_roberta-large-ca                  | diverse   |         505 |     0.2851 |        0.3133 |     0.3133 |            16657 |
| model_mono_ca_MrBERT-ca                         | diverse   |         505 |     0.2535 |        0.2665 |     0.2665 |            13948 |
| model_mono_gl_bertinho-gl                       | diverse   |         370 |     0.2405 |        0.254  |     0.254  |             7194 |
| model_mono_es_MrBERT-legal                      | diverse   |         405 |     0.237  |        0.2207 |     0.2207 |            14806 |
| model_mono_es_MrBERT-biomed                     | diverse   |         405 |     0.1951 |        0.1771 |     0.1771 |            11286 |
| model_mono_es_MrBERT-science                    | diverse   |         405 |     0.1506 |        0.1225 |     0.1225 |             8534 |
| model_mul_multilingual-e5-small                 | full      |        5000 |     0.9714 |        0.9805 |     0.6654 |            47416 |
| model_mul_potion-multilingual-128M              | full      |        5000 |     0.9674 |        0.9776 |     0.6786 |            39948 |
| model_mul_LaBSE                                 | full      |        5000 |     0.9646 |        0.9759 |     0.6271 |            47285 |
| model_mul_bert-base-multilingual-cased          | full      |        5000 |     0.9616 |        0.9745 |     0.6022 |            46756 |
| model_mul_paraphrase-multilingual-MiniLM-L12-v2 | full      |        5000 |     0.9548 |        0.9706 |     0.5569 |            27511 |
| model_mul_MrBERT                                | full      |        5000 |     0.9424 |        0.9603 |     0.4822 |            47072 |
| model_mono_pt_bertha-pt-small                   | full      |        5000 |     0.8278 |        0.8666 |     0.2334 |            44774 |
| model_mono_pt_bert-base-pt                      | full      |        5000 |     0.7774 |        0.8391 |     0.2287 |            50010 |
| model_mono_pt_bert-large-pt                     | full      |        5000 |     0.7654 |        0.8334 |     0.2405 |            49634 |
| model_mono_es_MrBERT-es                         | full      |        5000 |     0.7188 |        0.8001 |     0.1542 |            52078 |
| model_mono_es_MrBERT-legal                      | full      |        5000 |     0.4658 |        0.6098 |     0.1537 |            52987 |
| model_mono_ca_roberta-base-ca                   | full      |        5000 |     0.4286 |        0.5742 |     0.231  |            54602 |
| model_mono_ca_distilroberta-ca                  | full      |        5000 |     0.4326 |        0.5741 |     0.1999 |            35551 |
| model_mono_ca_MrBERT-ca                         | full      |        5000 |     0.3716 |        0.5196 |     0.2659 |            42454 |
| model_mono_ca_roberta-large-ca                  | full      |        5000 |     0.3584 |        0.501  |     0.1951 |            52661 |
| model_mono_es_MrBERT-science                    | full      |        5000 |     0.1926 |        0.301  |     0.1151 |            50198 |
| model_mono_gl_bertinho-gl                       | full      |        5000 |     0.1484 |        0.2479 |     0.1767 |            51257 |
| model_mono_es_MrBERT-biomed                     | full      |        5000 |     0.124  |        0.1965 |     0.136  |            52066 |

## Per-language weighted F1 (filtered from full/diverse)

### Dataset: diverse

| model                                           |     ca |     da |     de |     en |     es |     eu |     fr |     gl |     it |     nl |     pt |
|:------------------------------------------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| model_mono_ca_MrBERT-ca                         | 0.902  | 0.0391 | 0.04   | 0.1395 | 0.1241 | 0.1261 | 0.1429 | 0.1667 | 0.1884 | 0.0948 | 0.1139 |
| model_mono_ca_distilroberta-ca                  | 0.9647 | 0.0833 | 0.0739 | 0.1009 | 0.2128 | 0.2027 | 0.1905 | 0.2958 | 0.2101 | 0.0805 | 0.2484 |
| model_mono_ca_roberta-base-ca                   | 0.9773 | 0.0833 | 0.1208 | 0.105  | 0.2411 | 0.1802 | 0.2619 | 0.265  | 0.2174 | 0.1149 | 0.2502 |
| model_mono_ca_roberta-large-ca                  | 0.9647 | 0.0938 | 0.0539 | 0.0594 | 0.2014 | 0.1892 | 0.2143 | 0.2917 | 0.1739 | 0      | 0.2222 |
| model_mono_es_MrBERT-biomed                     | 0.1624 | 0.075  | 0.0833 | 0.0167 | 0.5024 | 0.0238 | 0.2292 | 0.2644 | 0      | 0.0792 | 0.1835 |
| model_mono_es_MrBERT-es                         | 0.1605 | 0.1183 | 0.1014 | 0.1708 | 0.9604 | 0.0529 | 0.1354 | 0.4138 | 0.0714 | 0.0278 | 0.2342 |
| model_mono_es_MrBERT-legal                      | 0.2058 | 0.0521 | 0.0884 | 0.0167 | 0.6367 | 0.0508 | 0.15   | 0.3172 | 0      | 0.1667 | 0.2521 |
| model_mono_es_MrBERT-science                    | 0.0754 | 0.0506 | 0.0739 | 0.0167 | 0.2701 | 0.019  | 0.0573 | 0.3473 | 0      | 0.0472 | 0.1774 |
| model_mono_gl_bertinho-gl                       | 0.0915 | 0.0481 | 0.0556 | 0.0476 | 0.2564 | 0.087  | 0.0827 | 1      | 0      | 0.0123 | 0.3918 |
| model_mono_pt_bert-base-pt                      | 0.1582 | 0.0425 | 0.0236 | 0.1244 | 0.1942 | 0.1154 | 0.1667 | 0.5935 | 0.0952 | 0.0444 | 0.9244 |
| model_mono_pt_bert-large-pt                     | 0.1191 | 0.0456 | 0.0417 | 0.077  | 0.1562 | 0.0641 | 0.1167 | 0.561  | 0.0952 | 0.0496 | 0.9259 |
| model_mono_pt_bertha-pt-small                   | 0.0739 | 0.0391 | 0.0542 | 0.1003 | 0.2052 | 0.1154 | 0.0625 | 0.5041 | 0.1786 | 0      | 0.9259 |
| model_mul_LaBSE                                 | 0.9349 | 0.9316 | 0.9708 | 0.9443 | 0.8757 | 0.9591 | 0.9423 | 0.9301 | 0.9078 | 0.9526 | 0.9338 |
| model_mul_MrBERT                                | 0.584  | 0.4427 | 0.5997 | 0.4849 | 0.604  | 0.5614 | 0.5321 | 0.5457 | 0.5234 | 0.4985 | 0.5393 |
| model_mul_bert-base-multilingual-cased          | 0.9474 | 0.8889 | 0.9686 | 0.9295 | 0.8359 | 0.9094 | 0.8782 | 0.9301 | 0.8936 | 0.8574 | 0.9325 |
| model_mul_multilingual-e5-small                 | 0.9574 | 0.8462 | 0.9331 | 0.9211 | 0.8561 | 0.8947 | 0.8385 | 0.9301 | 0.8723 | 0.9069 | 0.944  |
| model_mul_paraphrase-multilingual-MiniLM-L12-v2 | 0.8599 | 0.8376 | 0.8953 | 0.9323 | 0.8638 | 0.8509 | 0.816  | 0.8817 | 0.8865 | 0.8543 | 0.9206 |
| model_mul_potion-multilingual-128M              | 0.9489 | 0.9402 | 0.9748 | 0.9549 | 0.9309 | 0.9123 | 0.9487 | 0.9301 | 0.8936 | 0.929  | 0.9263 |

### Dataset: full

| model                                           |     ca |     da |     de |     en |     es |     eu |     fr |     gl |     it |     nl |     pt |
|:------------------------------------------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| model_mono_ca_MrBERT-ca                         | 0.9972 | 0.243  | 0.0652 | 0.5166 | 0.1278 | 0.1667 | 0.645  | 0.0698 | 0.4913 | 0.2567 | 0.4439 |
| model_mono_ca_distilroberta-ca                  | 0.9972 | 0.2203 | 0.1288 | 0.5852 | 0.2625 | 0      | 0.6364 | 0.1628 | 0.3243 | 0.3294 | 0.3856 |
| model_mono_ca_roberta-base-ca                   | 0.9972 | 0.1712 | 0.1696 | 0.5827 | 0.2187 | 0      | 0.6061 | 0.1081 | 0.2489 | 0.3002 | 0.3311 |
| model_mono_ca_roberta-large-ca                  | 0.9972 | 0.1577 | 0.0759 | 0.5012 | 0.2424 | 0      | 0.6364 | 0.0698 | 0.3243 | 0.2941 | 0.326  |
| model_mono_es_MrBERT-biomed                     | 0.1912 | 0.2167 | 0.4726 | 0.1862 | 0.8303 | 0.125  | 0.3409 | 0.5628 | 0.2857 | 0.1786 | 0.6085 |
| model_mono_es_MrBERT-es                         | 0.2082 | 0.2125 | 0.2696 | 0.853  | 0.9778 | 0.125  | 0.3939 | 0.5495 | 0.402  | 0.1492 | 0.6204 |
| model_mono_es_MrBERT-legal                      | 0.3429 | 0.2667 | 0.3392 | 0.6346 | 0.9    | 0.2917 | 0.5273 | 0.6907 | 0.3036 | 0.181  | 0.508  |
| model_mono_es_MrBERT-science                    | 0.1064 | 0.3292 | 0.3726 | 0.3044 | 0.7761 | 0.125  | 0.3052 | 0.56   | 0.3968 | 0.1706 | 0.5628 |
| model_mono_gl_bertinho-gl                       | 0.0753 | 0.0737 | 0.0373 | 0.2492 | 0.4449 | 0      | 0.1059 | 1      | 0      | 0      | 0.4929 |
| model_mono_pt_bert-base-pt                      | 0.1035 | 0.0966 | 0.2352 | 0.892  | 0.2261 | 0.5    | 0.4265 | 0.5162 | 0.3429 | 0.4725 | 0.9909 |
| model_mono_pt_bert-large-pt                     | 0.0733 | 0.1271 | 0.1745 | 0.8863 | 0.221  | 0.5    | 0.2834 | 0.4376 | 0.4405 | 0.3796 | 0.9909 |
| model_mono_pt_bertha-pt-small                   | 0.0379 | 0.0776 | 0.2052 | 0.9256 | 0.2249 | 0.5    | 0.2254 | 0.4227 | 0.2667 | 0.4126 | 1      |
| model_mul_LaBSE                                 | 0.9614 | 0.8565 | 0.7159 | 0.9843 | 0.8182 | 0.6    | 0.54   | 1      | 0.8095 | 0.6923 | 0.8674 |
| model_mul_MrBERT                                | 0.9405 | 0.7379 | 0.5603 | 0.9748 | 0.7332 | 0.6    | 0.2833 | 0.9452 | 0.8095 | 0.5128 | 0.7226 |
| model_mul_bert-base-multilingual-cased          | 0.9585 | 0.7908 | 0.7126 | 0.9827 | 0.8788 | 0.6    | 0.5167 | 1      | 0.9127 | 0.641  | 0.8419 |
| model_mul_multilingual-e5-small                 | 0.9614 | 0.8069 | 0.7066 | 0.9885 | 0.8678 | 0.6    | 0.54   | 1      | 0.8571 | 0.6667 | 0.8222 |
| model_mul_paraphrase-multilingual-MiniLM-L12-v2 | 0.9607 | 0.6284 | 0.6076 | 0.9798 | 0.8485 | 0.6    | 0.5667 | 1      | 0.8571 | 0.5385 | 0.8118 |
| model_mul_potion-multilingual-128M              | 0.9745 | 0.9569 | 0.8462 | 0.9843 | 0.8939 | 0.8    | 0.71   | 1      | 0.8571 | 0.8154 | 0.8781 |

## Per-language weighted F1 (dedicated by_lang/ CSVs)

| model                                           |     ca |     da |     de |     en |     es |     eu |     fr |     gl |     it |     nl |     pt |
|:------------------------------------------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| model_mono_ca_MrBERT-ca                         | 0.9943 | 0.1546 | 0.1408 | 0.523  | 0.3667 | 0.0711 | 0.4706 | 0.1174 | 0.3637 | 0.2834 | 0.3993 |
| model_mono_ca_distilroberta-ca                  | 0.995  | 0.2155 | 0.2473 | 0.5919 | 0.3919 | 0.1553 | 0.4458 | 0.1551 | 0.3766 | 0.2384 | 0.4223 |
| model_mono_ca_roberta-base-ca                   | 0.9952 | 0.1917 | 0.2081 | 0.5903 | 0.3439 | 0.1687 | 0.4229 | 0.1131 | 0.3965 | 0.2338 | 0.3659 |
| model_mono_ca_roberta-large-ca                  | 0.9956 | 0.1782 | 0.1667 | 0.5008 | 0.3531 | 0.1514 | 0.4653 | 0.1261 | 0.3585 | 0.2135 | 0.3381 |
| model_mono_es_MrBERT-biomed                     | 0.2255 | 0.2536 | 0.3656 | 0.1895 | 0.8869 | 0.0732 | 0.4195 | 0.5789 | 0.3805 | 0.2846 | 0.4877 |
| model_mono_es_MrBERT-es                         | 0.2653 | 0.3125 | 0.2792 | 0.8664 | 0.9823 | 0.1016 | 0.4844 | 0.5919 | 0.4288 | 0.2708 | 0.5425 |
| model_mono_es_MrBERT-legal                      | 0.3459 | 0.3097 | 0.3988 | 0.6475 | 0.9121 | 0.113  | 0.4894 | 0.6695 | 0.39   | 0.3111 | 0.5129 |
| model_mono_es_MrBERT-science                    | 0.1366 | 0.239  | 0.2432 | 0.3309 | 0.8008 | 0.0416 | 0.3704 | 0.5608 | 0.3046 | 0.276  | 0.4772 |
| model_mono_gl_bertinho-gl                       | 0.0802 | 0.0454 | 0.0602 | 0.2469 | 0.3637 | 0.0622 | 0.1481 | 0.9954 | 0.0823 | 0.0967 | 0.475  |
| model_mono_pt_bert-base-pt                      | 0.0693 | 0.1483 | 0.1317 | 0.8846 | 0.2722 | 0.0567 | 0.4548 | 0.626  | 0.3717 | 0.2691 | 0.9789 |
| model_mono_pt_bert-large-pt                     | 0.0664 | 0.1429 | 0.1201 | 0.8745 | 0.2808 | 0.0754 | 0.4413 | 0.5753 | 0.4123 | 0.2893 | 0.9795 |
| model_mono_pt_bertha-pt-small                   | 0.0444 | 0.1316 | 0.1175 | 0.9207 | 0.248  | 0.0693 | 0.4156 | 0.4888 | 0.2582 | 0.2523 | 0.9805 |
| model_mul_LaBSE                                 | 0.9684 | 0.8074 | 0.8363 | 0.9861 | 0.8384 | 0.9382 | 0.6133 | 0.9925 | 0.7952 | 0.6572 | 0.8254 |
| model_mul_MrBERT                                | 0.9441 | 0.6809 | 0.7274 | 0.9774 | 0.7467 | 0.7131 | 0.3979 | 0.9548 | 0.6823 | 0.4807 | 0.7188 |
| model_mul_bert-base-multilingual-cased          | 0.964  | 0.754  | 0.8129 | 0.9837 | 0.8288 | 0.8999 | 0.4968 | 0.991  | 0.7551 | 0.5905 | 0.8137 |
| model_mul_multilingual-e5-small                 | 0.9663 | 0.765  | 0.8272 | 0.9922 | 0.8244 | 0.9107 | 0.5326 | 0.9934 | 0.7467 | 0.6049 | 0.8062 |
| model_mul_paraphrase-multilingual-MiniLM-L12-v2 | 0.9614 | 0.7209 | 0.7886 | 0.9841 | 0.8253 | 0.8887 | 0.5127 | 0.9874 | 0.7098 | 0.6036 | 0.797  |
| model_mul_potion-multilingual-128M              | 0.9711 | 0.8837 | 0.8873 | 0.9863 | 0.8749 | 0.9116 | 0.6849 | 0.9931 | 0.8263 | 0.7601 | 0.8576 |

### Samples evaluated per language

| model                                           |   ca |   da |   de |   en |   es |   eu |   fr |   gl |   it |   nl |   pt |
|:------------------------------------------------|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|-----:|
| model_mono_ca_MrBERT-ca                         | 5000 | 1815 | 2014 | 5000 | 2897 |  308 |  836 | 3456 | 1357 | 1222 | 3435 |
| model_mono_ca_distilroberta-ca                  | 5000 | 1815 | 2014 | 5000 | 2897 |  308 |  836 | 3456 | 1357 | 1222 | 3435 |
| model_mono_ca_roberta-base-ca                   | 5000 | 1815 | 2014 | 5000 | 2897 |  308 |  836 | 3456 | 1357 | 1222 | 3435 |
| model_mono_ca_roberta-large-ca                  | 5000 | 1815 | 2014 | 5000 | 2897 |  308 |  836 | 3456 | 1357 | 1222 | 3435 |
| model_mono_es_MrBERT-biomed                     | 5000 | 1803 | 1995 | 5000 | 2942 |  322 |  840 | 3507 | 1334 | 1204 | 3469 |
| model_mono_es_MrBERT-es                         | 5000 | 1803 | 1995 | 5000 | 2942 |  322 |  840 | 3507 | 1334 | 1204 | 3469 |
| model_mono_es_MrBERT-legal                      | 5000 | 1803 | 1995 | 5000 | 2942 |  322 |  840 | 3507 | 1334 | 1204 | 3469 |
| model_mono_es_MrBERT-science                    | 5000 | 1803 | 1995 | 5000 | 2942 |  322 |  840 | 3507 | 1334 | 1204 | 3469 |
| model_mono_gl_bertinho-gl                       | 5000 | 1750 | 1950 | 5000 | 2813 |  335 |  759 | 3523 | 1464 | 1100 | 3393 |
| model_mono_pt_bert-base-pt                      | 5000 | 1845 | 2182 | 5000 | 2893 |  330 |  875 | 3503 | 1587 | 1225 | 4797 |
| model_mono_pt_bert-large-pt                     | 5000 | 1845 | 2182 | 5000 | 2893 |  330 |  875 | 3503 | 1587 | 1225 | 4797 |
| model_mono_pt_bertha-pt-small                   | 5000 | 1845 | 2182 | 5000 | 2893 |  330 |  875 | 3503 | 1587 | 1225 | 4797 |
| model_mul_LaBSE                                 | 5000 | 1883 | 2308 | 5000 | 3010 |  393 |  900 | 3590 | 1656 | 1274 | 4856 |
| model_mul_MrBERT                                | 5000 | 1883 | 2308 | 5000 | 3010 |  393 |  900 | 3590 | 1656 | 1274 | 4856 |
| model_mul_bert-base-multilingual-cased          | 5000 | 1883 | 2308 | 5000 | 3010 |  393 |  900 | 3590 | 1656 | 1274 | 4856 |
| model_mul_multilingual-e5-small                 | 5000 | 1883 | 2308 | 5000 | 3010 |  393 |  900 | 3590 | 1656 | 1274 | 4856 |
| model_mul_paraphrase-multilingual-MiniLM-L12-v2 | 5000 | 1883 | 2308 | 5000 | 3010 |  393 |  900 | 3590 | 1656 | 1274 | 4856 |
| model_mul_potion-multilingual-128M              | 5000 | 1883 | 2308 | 5000 | 3010 |  393 |  900 | 3590 | 1656 | 1274 | 4856 |

