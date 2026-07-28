# Monolingual vs Multilingual – Intent Classification

**Languages:** ca, da, de, es, gl, pt

**Dataset balance:** min_samples=10, max_samples=800

**Raspberry Pi budget:** ≤ 100.0 MB, within 5% of best F1

## Recommendations

| Language   | Best (unconstrained)                    |   Best F1 |   Best size (MB) | Best for RPi                         |   RPi F1 |   RPi size (MB) |
|:-----------|:----------------------------------------|----------:|-----------------:|:-------------------------------------|---------:|----------------:|
| ca         | roberta-base-ca                         |    0.9759 |             56.2 | MrBERT-ca                            |   0.9725 |            56.2 |
| da         | potion-multilingual-128M (multilingual) |    0.923  |            518.5 | LaBSE (multilingual)                 |   0.8737 |           518.5 |
| de         | potion-multilingual-128M (multilingual) |    0.9215 |            518.5 | multilingual-e5-small (multilingual) |   0.8799 |           262.1 |
| es         | MrBERT-es                               |    0.9485 |             55.8 | MrBERT-es                            |   0.9485 |            55.8 |
| gl         | potion-multilingual-128M (multilingual) |    0.9849 |            518.5 | bertinho-gl                          |   0.9707 |            35.2 |
| pt         | bert-base-pt                            |    0.9577 |             35.5 | bert-base-pt                         |   0.9577 |            35.5 |

## Full results by language

### [CA]

| Model                                                | Type         |   Accuracy |       F1 |   Size (MB) | Params   |   Throughput (sps) |
|:-----------------------------------------------------|:-------------|-----------:|---------:|------------:|:---------|-------------------:|
| roberta-base-ca                                      | monolingual  |   0.976462 | 0.975898 |     56.2441 | 12.97M   |            38703.2 |
| distilroberta-ca                                     | monolingual  |   0.975511 | 0.974764 |     56.2441 | 12.97M   |            35527.6 |
| roberta-large-ca                                     | monolingual  |   0.97456  | 0.974281 |     56.2441 | 12.97M   |            39404.7 |
| MrBERT-ca                                            | monolingual  |   0.973134 | 0.972495 |     56.1802 | 12.94M   |            37183.4 |
| potion-multilingual-128M (multilingual)              | multilingual |   0.949834 | 0.94234  |    518.495  | 129.09M  |            55143.4 |
| LaBSE (multilingual)                                 | multilingual |   0.949834 | 0.941353 |    518.487  | 129.27M  |            51570.1 |
| bert-base-multilingual-cased (multilingual)          | multilingual |   0.943176 | 0.93242  |    128.158  | 30.82M   |            49848.6 |
| multilingual-e5-small (multilingual)                 | multilingual |   0.942463 | 0.930796 |    262.052  | 64.50M   |            33873.5 |
| paraphrase-multilingual-MiniLM-L12-v2 (multilingual) | multilingual |   0.932715 | 0.922507 |    262.052  | 64.50M   |            38956.4 |
| MrBERT (multilingual)                                | multilingual |   0.882311 | 0.87379  |    267.442  | 66.01M   |            52796.6 |

### [DA]

| Model                                                | Type         |   Accuracy |       F1 |   Size (MB) | Params   |   Throughput (sps) |
|:-----------------------------------------------------|:-------------|-----------:|---------:|------------:|:---------|-------------------:|
| potion-multilingual-128M (multilingual)              | multilingual |   0.912688 | 0.922973 |     518.495 | 129.09M  |            53821.5 |
| LaBSE (multilingual)                                 | multilingual |   0.872442 | 0.873727 |     518.487 | 129.27M  |            52928   |
| multilingual-e5-small (multilingual)                 | multilingual |   0.858117 | 0.852047 |     262.052 | 64.50M   |            55993.2 |
| bert-base-multilingual-cased (multilingual)          | multilingual |   0.851296 | 0.842996 |     128.158 | 30.82M   |            52734.2 |
| paraphrase-multilingual-MiniLM-L12-v2 (multilingual) | multilingual |   0.840382 | 0.834506 |     262.052 | 64.50M   |            42282.4 |
| MrBERT (multilingual)                                | multilingual |   0.75648  | 0.750764 |     267.442 | 66.01M   |            54721   |

### [DE]

| Model                                                | Type         |   Accuracy |       F1 |   Size (MB) | Params   |   Throughput (sps) |
|:-----------------------------------------------------|:-------------|-----------:|---------:|------------:|:---------|-------------------:|
| potion-multilingual-128M (multilingual)              | multilingual |   0.91931  | 0.92151  |     518.495 | 129.09M  |            58593.5 |
| LaBSE (multilingual)                                 | multilingual |   0.897051 | 0.894583 |     518.487 | 129.27M  |            23230.7 |
| multilingual-e5-small (multilingual)                 | multilingual |   0.885921 | 0.879907 |     262.052 | 64.50M   |            55704.2 |
| bert-base-multilingual-cased (multilingual)          | multilingual |   0.875904 | 0.869689 |     128.158 | 30.82M   |            55861.8 |
| paraphrase-multilingual-MiniLM-L12-v2 (multilingual) | multilingual |   0.852532 | 0.847374 |     262.052 | 64.50M   |            36712.6 |
| MrBERT (multilingual)                                | multilingual |   0.770173 | 0.763289 |     267.442 | 66.01M   |            46863.1 |

### [ES]

| Model                                                | Type         |   Accuracy |       F1 |   Size (MB) | Params   |   Throughput (sps) |
|:-----------------------------------------------------|:-------------|-----------:|---------:|------------:|:---------|-------------------:|
| MrBERT-es                                            | monolingual  |   0.94854  | 0.948481 |     55.7828 | 12.92M   |            37188.1 |
| potion-multilingual-128M (multilingual)              | multilingual |   0.910987 | 0.908405 |    518.495  | 129.09M  |            43801.3 |
| LaBSE (multilingual)                                 | multilingual |   0.902643 | 0.891402 |    518.487  | 129.27M  |            38440.8 |
| bert-base-multilingual-cased (multilingual)          | multilingual |   0.890125 | 0.878077 |    128.158  | 30.82M   |            45689.3 |
| multilingual-e5-small (multilingual)                 | multilingual |   0.888271 | 0.872981 |    262.052  | 64.50M   |            42468.5 |
| paraphrase-multilingual-MiniLM-L12-v2 (multilingual) | multilingual |   0.880853 | 0.870654 |    262.052  | 64.50M   |            35006.6 |
| MrBERT-legal                                         | monolingual  |   0.874363 | 0.859276 |    265.782  | 66.01M   |            36962   |
| MrBERT-biomed                                        | monolingual  |   0.839128 | 0.823975 |    265.782  | 66.01M   |            35226.2 |
| MrBERT (multilingual)                                | multilingual |   0.805285 | 0.789511 |    267.442  | 66.01M   |            45255.8 |
| MrBERT-science                                       | monolingual  |   0.748725 | 0.70333  |    265.782  | 66.01M   |            33608   |

### [GL]

| Model                                                | Type         |   Accuracy |       F1 |   Size (MB) | Params   |   Throughput (sps) |
|:-----------------------------------------------------|:-------------|-----------:|---------:|------------:|:---------|-------------------:|
| potion-multilingual-128M (multilingual)              | multilingual |   0.983891 | 0.984946 |    518.495  | 129.09M  |            40892.8 |
| multilingual-e5-small (multilingual)                 | multilingual |   0.981413 | 0.982301 |    262.052  | 64.50M   |            42540.1 |
| LaBSE (multilingual)                                 | multilingual |   0.981413 | 0.981958 |    518.487  | 129.27M  |            36547.1 |
| bert-base-multilingual-cased (multilingual)          | multilingual |   0.977076 | 0.978113 |    128.158  | 30.82M   |            45240.7 |
| paraphrase-multilingual-MiniLM-L12-v2 (multilingual) | multilingual |   0.973358 | 0.974671 |    262.052  | 64.50M   |            30867.1 |
| bertinho-gl                                          | monolingual  |   0.972739 | 0.970692 |     35.2011 | 7.74M    |            34216.5 |
| MrBERT (multilingual)                                | multilingual |   0.92627  | 0.922701 |    267.442  | 66.01M   |            37610.8 |

### [PT]

| Model                                                | Type         |   Accuracy |       F1 |   Size (MB) | Params   |   Throughput (sps) |
|:-----------------------------------------------------|:-------------|-----------:|---------:|------------:|:---------|-------------------:|
| bert-base-pt                                         | monolingual  |   0.958567 | 0.957729 |     35.4699 | 7.66M    |            27780.8 |
| bertha-pt-small                                      | monolingual  |   0.95782  | 0.95697  |     35.8302 | 7.74M    |            28717.4 |
| bert-large-pt                                        | monolingual  |   0.957074 | 0.955804 |     35.4699 | 7.66M    |            29570.1 |
| potion-multilingual-128M (multilingual)              | multilingual |   0.922732 | 0.916859 |    518.495  | 129.09M  |            40405.2 |
| LaBSE (multilingual)                                 | multilingual |   0.916387 | 0.903534 |    518.487  | 129.27M  |            46936.1 |
| multilingual-e5-small (multilingual)                 | multilingual |   0.910788 | 0.896347 |    262.052  | 64.50M   |            44215.8 |
| bert-base-multilingual-cased (multilingual)          | multilingual |   0.905189 | 0.894614 |    128.158  | 30.82M   |            40865   |
| paraphrase-multilingual-MiniLM-L12-v2 (multilingual) | multilingual |   0.891377 | 0.8781   |    262.052  | 64.50M   |            37405.2 |
| MrBERT (multilingual)                                | multilingual |   0.788354 | 0.772295 |    267.442  | 66.01M   |            42168.7 |

