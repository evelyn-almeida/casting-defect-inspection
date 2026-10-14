|                                       |   ROC AUC |   Accuracy |   Defect recall |   Good-part recall |     F1 |   Threshold |
|:--------------------------------------|----------:|-----------:|----------------:|-------------------:|-------:|------------:|
| HOG + LBP + logistic regression       |    0.9906 |     0.8406 |          0.9978 |             0.5687 | 0.888  |        0.11 |
| CNN (standard augmentation)           |    0.9957 |     0.9762 |          0.9934 |             0.9466 | 0.9815 |        0.24 |
| CNN (+ noise & lighting augmentation) |    0.9967 |     0.9301 |          0.9934 |             0.8206 | 0.9474 |        0.66 |