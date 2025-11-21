import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import os

# GroundingDINOのインポート（公式ライブラリを使用）
GROUNDINGDINO_AVAILABLE = False
try:
    from groundingdino.util.inference import load_model, predict
    import groundingdino.datasets.transforms as T
    GROUNDINGDINO_AVAILABLE = True
except ImportError:
    try:
        # 別のインポート方法を試す
        import sys
        # GroundingDINOがインストールされている場合のパスを追加
        groundingdino_path = os.path.join(os.path.dirname(__file__), "GroundingDINO")
        if os.path.exists(groundingdino_path):
            sys.path.insert(0, groundingdino_path)
        from groundingdino.util.inference import load_model, predict
        import groundingdino.datasets.transforms as T
        GROUNDINGDINO_AVAILABLE = True
    except ImportError:
        pass


def get_device():
    """利用可能なデバイスを自動検出する"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        # MPS (Metal Performance Shaders) for macOS
        return "mps" # 対応が不十分なおそれがあるため、不具合が発生した場合、"cpu"に置き換えること
    else:
        return "cpu"


def transform_image(image_pil: Image.Image):
    """PIL画像をGroundingDINO入力用に変換する"""
    import groundingdino.datasets.transforms as T
    
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image



@st.cache_resource
def load_groundingdino_model(model_path: str, config_path: str = None):
    """GroundingDINOモデルを読み込む"""
    if not GROUNDINGDINO_AVAILABLE:
        return None, None
    
    try:
        # デフォルトの設定パス
        if config_path is None:
            # 一般的な設定ファイルのパスを試す
            possible_configs = [
                "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "groundingdino/config/GroundingDINO_SwinT_OGC.py",
                "config/GroundingDINO_SwinT_OGC.py",
            ]
            config_path = None
            for path in possible_configs:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                # 設定ファイルが見つからない場合、デフォルト設定を使用
                st.warning("設定ファイルが見つかりません。デフォルト設定を使用します。")
                # モデルファイルから設定を推測
                config_path = "GroundingDINO_SwinT_OGC.py"  # 相対パス
        
        # モデルの読み込み
        # load_modelの定義によってはdevice引数がない場合もあるため、まずは読み込んでからto(device)する
        model = load_model(config_path, model_path)
        
        device = get_device()
        model = model.to(device)
        model.eval()
        
        return model, config_path
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {str(e)}")
        st.info("GroundingDINOをインストールしてください: pip install git+https://github.com/IDEA-Research/GroundingDINO.git")
        return None, None


def draw_bounding_boxes(image: Image.Image, boxes, phrases, logits=None, box_threshold: float = 0.3):
    """画像にバウンディングボックスを描画する（Pillow使用）"""
    if boxes is None or len(boxes) == 0:
        return image
    
    # 画像のコピーを作成
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # フォントの設定
    try:
        # macOS用
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
    except:
        try:
            # Windows用
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            # デフォルトフォント
            font = ImageFont.load_default()
    
    # バウンディングボックスを描画
    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan", "magenta"]
    
    # logitsがNoneの場合はダミーのリストを作成
    if logits is None:
        logits = [None] * len(boxes)
    
    for i, (box, phrase, logit) in enumerate(zip(boxes, phrases, logits)):
        # ボックス座標を取得
        if isinstance(box, (list, tuple, np.ndarray)):
            if len(box) == 4:
                x1, y1, x2, y2 = box
            else:
                continue
        else:
            continue
        
        # 座標を整数に変換
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 色を選択
        color = colors[i % len(colors)]
        
        # 矩形を描画
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # ラベルを描画
        if logit is not None:
            label = f"{phrase} {logit:.2f}"
        else:
            label = f"{phrase}"
            
        # テキストのバウンディングボックスを取得
        try:
            bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        except:
            # textbboxが使えない場合は手動で計算
            text_width = len(label) * 8
            text_height = 16
            bbox = (x1, y1 - text_height - 4, x1 + text_width, y1)
        
        # テキストの背景を描画
        draw.rectangle(bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill="white", font=font)
    
    return draw_image


def main():
    st.set_page_config(page_title="GroundingDINO オブジェクト検出", layout="wide")
    st.title("GroundingDINO オブジェクト検出アプリ")
    st.markdown("画像とプロンプトを入力すると、検出されたオブジェクトにバウンディングボックスを描画します。")
    
    # GroundingDINOの利用可能性をチェック
    if not GROUNDINGDINO_AVAILABLE:
        st.error("GroundingDINOライブラリが見つかりません。")
        st.info("""
        **インストール方法:**
        1. GitHubリポジトリをクローン:
           ```bash
           git clone https://github.com/IDEA-Research/GroundingDINO.git
           cd GroundingDINO
           pip install -e .
           ```
        2. または、pipで直接インストール:
           ```bash
           pip install git+https://github.com/IDEA-Research/GroundingDINO.git
           ```
        """)
        st.stop()
    
    # モデルの読み込み
    model_path = "models/groundingdino_swint_ogc.pth"
    
    if not Path(model_path).exists():
        st.error(f"モデルファイルが見つかりません: {model_path}")
        st.stop()
    
    model, config_path = load_groundingdino_model(model_path)
    
    if model is None:
        st.error("モデルの読み込みに失敗しました。")
        st.info("""
        **トラブルシューティング:**
        - GroundingDINOの設定ファイル（config）が必要です
        - 設定ファイルは通常、GroundingDINOリポジトリ内の `groundingdino/config/GroundingDINO_SwinT_OGC.py` にあります
        - リポジトリをクローンして、設定ファイルへのパスを確認してください
        """)
        st.stop()
    
    # サイドバー
    with st.sidebar:
        st.header("設定")
        box_threshold = st.slider("ボックス閾値", 0.0, 1.0, 0.3, 0.05)
        text_threshold = st.slider("テキスト閾値", 0.0, 1.0, 0.25, 0.05)
        st.markdown("---")
        st.info(f"設定ファイル: {config_path}")
    
    # メインコンテンツ
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("入力")
        uploaded_file = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])
        text_prompt = st.text_input("プロンプト", placeholder="例: dog . cat . person", help="検出したいオブジェクトを記述してください。複数の場合は '.' で区切ります。")
        
        if uploaded_file is not None:
            # 画像を読み込み
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="アップロードされた画像", use_container_width=True)
            
            if text_prompt:
                # 推論実行
                if st.button("検出実行", type="primary"):
                    with st.spinner("オブジェクトを検出中..."):
                        try:
                            # 画像を前処理
                            image_tensor = transform_image(image)
                            
                            # 推論実行
                            device = get_device()
                            boxes, logits, phrases = predict(
                                model=model,
                                image=image_tensor,
                                caption=text_prompt,
                                box_threshold=box_threshold,
                                text_threshold=text_threshold,
                                device=device
                            )

                            print("boxes", boxes, "logits", logits, "phrases", phrases)
                            
                            if boxes is not None and len(boxes) > 0:
                                # Tensorをnumpy配列に変換
                                boxes = boxes.cpu().numpy() if torch.is_tensor(boxes) else boxes
                                logits = logits.cpu().numpy() if torch.is_tensor(logits) else logits
                                
                                # 座標変換: [cx, cy, w, h] (正規化) -> [x1, y1, x2, y2] (絶対座標)
                                W, H = image.size
                                boxes_xyxy = []
                                for box in boxes:
                                    cx, cy, w, h = box
                                    x1 = (cx - w / 2) * W
                                    y1 = (cy - h / 2) * H
                                    x2 = (cx + w / 2) * W
                                    y2 = (cy + h / 2) * H
                                    boxes_xyxy.append([x1, y1, x2, y2])
                                
                                # バウンディングボックスを描画
                                result_image = draw_bounding_boxes(image, boxes_xyxy, phrases, logits, box_threshold)
                                
                                with col2:
                                    st.header("結果")
                                    st.image(result_image, caption="検出結果", use_container_width=True)
                                    st.success(f"{len(boxes)}個のオブジェクトを検出しました")
                                    
                                    # 検出結果の詳細
                                    with st.expander("検出結果の詳細"):
                                        for i, (box, phrase, logit) in enumerate(zip(boxes_xyxy, phrases, logits)):
                                            st.write(f"**オブジェクト {i+1}**: {phrase}")
                                            st.write(f"信頼度: {logit:.3f}")
                                            st.write(f"座標: ({box[0]:.2f}, {box[1]:.2f}) - ({box[2]:.2f}, {box[3]:.2f})")
                            else:
                                st.warning("オブジェクトが検出されませんでした。プロンプトや閾値を調整してみてください。")
                        except Exception as e:
                            st.error(f"推論中にエラーが発生しました: {str(e)}")
                            st.exception(e)
            else:
                st.info("プロンプトを入力してください。")
        else:
            st.info("画像をアップロードしてください。")


if __name__ == "__main__":
    main()
