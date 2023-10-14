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

## 程式碼分割

以下會將未來程式可能的結構切分開來，方便大家各自處理。同時也會對相互之間的 Interface 進行規範，請遵守以利後續對接。

### Input

下載到`./input`並讀取指定輸入檔案

輸出：

```
@dataclass
Class Dataset{
  x_train: pd.DataFrame
  x_val: pd.DataFrame
  y_train: pd.DataFrame
  y_val: pd.DataFrame
  x_test: pd.DataFrame
}
```

### Output

將預測輸出成指定格式到`./output/`

輸入：

```
// 對 Popularity 的預測
y_pred: np.numpy
```
