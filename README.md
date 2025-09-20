# テキスト分析アプリ

このアプリは、テキストデータを分析してワードクラウドと共起ネットワークを生成するStreamlitアプリケーションです。

## 機能

- 📊 テキストデータの形態素解析
- 🔤 ワードクラウド生成
- 🕸️ 共起ネットワーク可視化
- 📈 中心性分析
- 💾 結果のダウンロード

## 使用方法

### ローカル実行

1. 必要なパッケージをインストール：
```bash
pip install -r requirements.txt
```

2. アプリを実行：
```bash
streamlit run app.py
```

### Streamlit Cloud デプロイ

1. このリポジトリをGitHubにプッシュ
2. [Streamlit Cloud](https://share.streamlit.io/)でアプリをデプロイ
3. リポジトリURLを指定してデプロイ

## ファイル構成

```
├── app.py                    # メインアプリケーション
├── requirements.txt          # Python依存関係
├── .streamlit/
│   └── config.toml          # Streamlit設定
├── csv/
│   └── data.csv             # 分析対象データ
├── font/
│   └── NotoSansJP-VariableFont_wght.ttf  # 日本語フォント
└── lib/                     # 静的ファイル（pyvis用）
    ├── bindings/
    ├── tom-select/
    └── vis-9.1.2/
```

## 必要なデータ形式

CSVファイルに`text`列が含まれている必要があります：

```csv
投稿日時,text
2025/09/06 19:21,テキスト内容1
2025/09/06 13:20,テキスト内容2
```

## 技術スタック

- **Streamlit**: Webアプリケーションフレームワーク
- **spaCy + GINZA**: 日本語形態素解析
- **WordCloud**: ワードクラウド生成
- **NetworkX**: ネットワーク分析
- **Plotly**: インタラクティブ可視化
- **pyvis**: 高度なネットワーク可視化
- **Matplotlib**: 静的グラフ生成

## 注意事項

- Streamlit Cloudでは日本語フォントが利用できない場合があります
- 大量のデータを処理する場合は、メモリ使用量にご注意ください
- 一時ファイルは自動的にクリーンアップされます