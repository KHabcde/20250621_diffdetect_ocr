# main.py

import os
import cv2
from ocr_utils import (
    preprocess_image,
    analyze_with_tesseract,
    generate_candidate_phrases,
    find_best_match,
    convert_to_global_coords
)

def find_text_coordinates(cropped_image_path: str, crop_area_vertices: tuple, target_text: str):
    """
    画像とターゲットテキストから、そのテキストの全体座標を推定するモジュール。

    Args:
        cropped_image_path (str): 切り出した画像のファイルパス。
        crop_area_vertices (tuple): スクリーンショット内での切り出し領域の座標 (左上x, 左上y, 右下x, 右下y)。
        target_text (str): 座標を特定したいテキスト。

    Returns:
        dict: 検出されたテキストの中心座標 {'x': int, 'y': int} とバウンディングボックス {'bbox': [x1,y1,x2,y2]} を含む辞書。
              見つからない場合は None。
    """
    print(f"--- ターゲット「{target_text}」の座標推定を開始 ---")

    # 1. 画像読み込みと前処理
    cropped_image = cv2.imread(cropped_image_path)
    if cropped_image is None:
        print(f"エラー: 画像 '{cropped_image_path}' が読み込めません。")
        return None
    
    processed_image = preprocess_image(cropped_image)
    
    # 2. OCR実行
    print("OCRを実行中...")
    ocr_results = analyze_with_tesseract(processed_image)
    if not ocr_results:
        print("OCRでテキストが検出されませんでした。")
        return None

    # 3. 候補フレーズ生成
    print("候補フレーズを生成中...")
    candidates = generate_candidate_phrases(ocr_results)
    
    # デバッグ用に候補を一部表示
    # print("\n--- 生成された候補（一部）---")
    # for c in candidates[:10]: print(f"  - '{c['text']}'")
    # print("-" * 20)

    # 4. 最適な候補を検索
    print("最適な候補を検索中...")
    best_match = find_best_match(target_text, candidates)
    
    # 5. 結果の評価と座標変換
    # スコアの閾値を設定 (0.7は70%のマッチ度。要調整)
    SCORE_THRESHOLD = 0.7 
    
    if best_match and best_match['score'] > SCORE_THRESHOLD:
        print("\n[成功] ターゲットに一致する候補が見つかりました！")
        print(f"  - 検出テキスト: '{best_match['text']}'")
        print(f"  - マッチスコア: {best_match['score']:.2f}")

        crop_origin = (crop_area_vertices[0], crop_area_vertices[1])
        global_bbox = convert_to_global_coords(best_match['bbox'], crop_origin)
        
        center_x = global_bbox[0] + (global_bbox[2] - global_bbox[0]) // 2
        center_y = global_bbox[1] + (global_bbox[3] - global_bbox[1]) // 2

        result = {
            "center": {"x": center_x, "y": center_y},
            "bbox": global_bbox,
            "text": best_match['text'],
            "score": best_match['score']
        }
        print(f"  - 全体座標 (BBox): {result['bbox']}")
        print(f"  - 推定中心座標: {result['center']}")
        return result
    else:
        print("\n[失敗] ターゲットに一致する信頼性の高い候補が見つかりませんでした。")
        if best_match:
            print(f"  (最も近かった候補: '{best_match['text']}', スコア: {best_match['score']:.2f})")
        return None


# ==============================================================================
# --- このスクリプトを直接実行した場合のテストコード ---
# ==============================================================================
if __name__ == "__main__":
    
    # --- ユーザー設定 ---
    # ご自身の切り出し画像と座標に合わせて設定してください
    # このサンプル画像がない場合、エラーになります。
    CROPPED_FILE_PATH = "macro_diff_region.png" # ここにご自身の画像パスを指定
    
    # スクリーンショット内での切り出し領域の頂点座標 (左上x, 左上y, 右下x, 右下y)
    CROP_AREA_VERTICES = (936, 43, 1225, 530) # 画像に合わせて調整

    # --- テスト実行 ---
    if not os.path.exists(CROPPED_FILE_PATH):
        print(f"テストエラー: テスト画像 '{CROPPED_FILE_PATH}' が見つかりません。")
        print("ご自身の切り出し画像ファイルを指定してください。")
    else:
        # テストケース1
        target_coords = find_text_coordinates(
            cropped_image_path=CROPPED_FILE_PATH,
            crop_area_vertices=CROP_AREA_VERTICES,
            target_text="名前を付けて保存"
        )
        print("-" * 50)
        
        # テストケース2
        target_coords_2 = find_text_coordinates(
            cropped_image_path=CROPPED_FILE_PATH,
            crop_area_vertices=CROP_AREA_VERTICES,
            target_text="新しいウィンドウ"
        )
        print("-" * 50)

        # テストケース3（少し間違っていても見つかるか）
        target_coords_3 = find_text_coordinates(
            cropped_image_path=CROPPED_FILE_PATH,
            crop_area_vertices=CROP_AREA_VERTICES,
            target_text="ページ設定"
        )
        print("-" * 50)