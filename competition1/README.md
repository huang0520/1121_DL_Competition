# Competition 1 Predicting News Popularity

- [Tutorial Notebook](https://nthu-datalab.github.io/ml/competitions/Comp_01_Text-Feature-Engineering/01_Text_Feature_Engineering.html)
- [Kaggle](https://www.kaggle.com/competitions/2023-datalab-cup1-predicting-news-popularity)

## 特殊套件說明

因為本次競賽禁止使用 DL 相關的模型，因此我先安裝了三個 GBDT 的模型，以利後續使用：

- `XGBoost`
- `CatBoost`
- `lightGBM`

  關於 `lightGBM`，如果用 MAC 第一次安裝有可能會出問題，請去安裝 `cmake` 及 `libomp`。如果有使用 `Homebrew`，可以直接：

  - `brew install cmake libomp`
