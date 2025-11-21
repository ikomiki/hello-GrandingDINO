# GroundingDINO-hello

## 前提

- uv がインストール済み

## モデルのダウンロード

## 初期設定

```sh
# 仮想環境の作成と有効化
uv venv
source .venv/bin/activate

# まず、プロジェクトの依存関係（torchを含む）をインストール
uv pip install -e .

# GroundingDINOリポジトリをクローン
git clone https://github.com/IDEA-Research/GroundingDINO.git

# GroundingDINOをインストール（ビルド分離を無効化して現在の環境のtorchを使用）
uv pip install --no-build-isolation -e GroundingDINO
```

```sh
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O models/groundingdino_swint_ogc.pth
```

## 使用方法

```sh
# Streamlitアプリを起動
streamlit run main.py
```

## 注意事項

- GroundingDINO のインストールには torch が必要なため、先にプロジェクトの依存関係をインストールしてください
- モデルファイル（`models/groundingdino_swint_ogc.pth`）が必要です
