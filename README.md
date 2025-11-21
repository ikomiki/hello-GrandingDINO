# GroundingDINO-hello

GroundingDINOを使用した、テキストプロンプトベースの物体検出Streamlitアプリケーションです。

## 必要要件

- uv（Pythonパッケージマネージャー）

## セットアップ手順

### 1. リポジトリのクローン

```bash
git clone https://github.com/ikomiki/hello-GrandingDINO
cd hello-GrandingDINO
```

### 2. 依存関係のインストール

このプロジェクトは`uv`を使用して依存関係を管理しています。

```bash
# 依存関係のインストール
uv sync
```

### 3. GroundingDINOのインストール

```sh
# 環境
uv sync

# GroundingDINOリポジトリをクローン
git clone https://github.com/IDEA-Research/GroundingDINO.git

# GroundingDINOをインストール（ビルド分離を無効化して現在の環境のtorchを使用）
uv pip install --no-build-isolation -e GroundingDINO
```

### 4. モデルのダウンロード

```sh
mkdir -p models
curl -L https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
    -o models/groundingdino_swint_ogc.pth
```

### 5. アプリケーションの起動

```sh
# Streamlitアプリを起動
uv run streamlit run main.py
```

ブラウザが自動的に開き、アプリケーションが表示されます（通常は `http://localhost:8501`）。

## 注意事項

- GroundingDINO のインストールには torch が必要なため、先にプロジェクトの依存関係をインストールしてください
- モデルファイル（`models/groundingdino_swint_ogc.pth`）が必要です
