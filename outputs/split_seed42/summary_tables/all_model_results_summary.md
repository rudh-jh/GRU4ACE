| model_name | category | feature_input | selected_dim | sampling | backbone | ACC | Sn | Sp | MCC | F1 | AUC | AUPR | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GRU_EN306_SMOTE | mainline | BPF+FEGS+Fasttext+ProtT5+BERT+ESM2 | 306 | SMOTE | GRU | 0.843137 | 0.823684 | 0.854687 | 0.670241 | 0.796461 | 0.929400 | 0.893075 | 当前最佳主结果：EN-306 + SMOTE + GRU |
| GRU_EN306_RUS | mainline | BPF+FEGS+Fasttext+ProtT5+BERT+ESM2 | 306 | RUS | GRU | 0.836275 | 0.797368 | 0.859375 | 0.653053 | 0.783658 | 0.918565 | 0.877466 | 主线模型：EN-306 + RUS + GRU |
| BPF+FEGS_EN_LR | baseline | BPF+FEGS | 273 | None | Logistic Regression | 0.784314 | 0.736842 | 0.812500 | 0.544000 | 0.717949 | 0.860300 |  | 两特征 EN+LR 对照基线 |
| BPF+FEGS_MLP | baseline | BPF+FEGS | 1019 | None | MLP | 0.769608 | 0.644737 | 0.843750 | 0.499198 | 0.675862 | 0.853002 |  | 早期基线，不是最终主线 |
| BPF+FEGS_EN273_MLP | baseline | BPF+FEGS | 273 | None | MLP | 0.754902 | 0.723684 | 0.773438 | 0.488365 | 0.687500 | 0.848890 |  | 两特征早期 EN-273 基线，不是最终 306 主线 |