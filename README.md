# 1121_DL_Competition

---

以下都是個人目前的建議，如果兩位覺得 OK 的話，我們就照這個去執行。
如果有什麼想調整或建議，歡迎提出。

## 環境設置

為了統一大家的工作環境，這邊先建好一個基礎 Conda 及 Pip 的設置文件。由於 Server 掛了，如 PyTorch, Tensorflow 需要 CUDA 的就先沒裝，第一次應該也還不用，之後再調整。

使用說明：

1. 先使用 Conda 建立虛擬環境： `conda env create -f /path/to/environment.yml`
2. 啟動虛擬環境：`conda activate DL_Comp`
3. 安裝 Pip 環境：`pip install -r /path/to/requirement.txt`

## 規範

### 格式化規範

為了一定程度上的程式碼一致性，我希望統一使用 Black formatter 來格式化程式碼。Extension 的 Recommendation 我已經設置好，打開 `VSCode Marketplace` 應該就可以在推薦區看到。

而格式化的相關設定都已在 `Workspace` 的設定中設置好，如果想確認有什麼設定，請查看 `1121_DL_Competition/.vscode/settings.json`。

### 命名規範

同樣為了一致性，我希望能夠使用符合 PEP 規範的命名方式來對命名：

- `CONSTANT_NAME`: 希望不變的常數請使用 **" 大寫 + 底線 "**
- `ClassName`: Class 請使用 **" 首字大寫不加底線 "**
- `other_name`: 其餘的都請使用 **" 小寫 + 底線 "**

### Git 規範

為了方便管理，main 會被我鎖起來。請自行拉一個 Branch 出去，等到一個區塊完成的差不多再通過 PR merge 回 main。

### 註解建議

這部份不是規範，但我希望兩位要多寫點 Comment 方便之後大家去看 Code 的時候會比較容易點。然後 Jupyter 的 Markdown 功能可以多用，寫長串或是解釋整個 Block 會比 Comment 好用。

## 程式分割
