# ocr_utils.py

import cv2
import pytesseract
import Levenshtein # pip install python-levenshtein
import easyocr

# --- Tesseract OCR の設定 ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# EasyOCRリーダーは重いので、グローバルで一度だけ初期化する
EASYOCR_READER = None

def get_easyocr_reader():
    """EasyOCRリーダーをシングルトンとして取得する"""
    global EASYOCR_READER
    if EASYOCR_READER is None:
        print("EasyOCRのリーダーを初回初期化しています...")
        EASYOCR_READER = easyocr.Reader(['ja', 'en'], gpu=False)
    return EASYOCR_READER

def preprocess_image(image):
    """画像をグレースケール化し、拡大・二値化して返す"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # --- ↓ここから追加 ---
    # 元の画像の高さを取得
    h, w = gray.shape
    # 文字が小さいことを想定し、画像を2倍に拡大する
    # 補間方法は、品質と速度のバランスが良いINTER_CUBICを推奨
    gray_resized = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    # --- ↑ここまで追加 ---
    
    # 拡大した画像に対して二値化を行う
    _, binary_image = cv2.threshold(gray_resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    inverted_image = cv2.bitwise_not(binary_image)
    
    return inverted_image

def analyze_with_tesseract(image):
    """Tesseract OCRで画像を分析し、単語リストを返す"""
    # PSM 11: まばらなテキストとして認識。UI要素に適していることが多い
    config = '--psm 11 --oem 3'
    
    data = pytesseract.image_to_data(
        image, lang='jpn+eng', config=config, output_type=pytesseract.Output.DICT
    )
    
    results = []
    for i in range(len(data['text'])):
        # 信頼度が低いものや空のテキストは除外
        if int(data['conf'][i]) > 30 and data['text'][i].strip() != "":
            # 2倍に拡大したので、座標とサイズを1/2に戻す
            x = data['left'][i] // 2
            y = data['top'][i] // 2
            w = data['width'][i] // 2
            h = data['height'][i] // 2
            results.append({
                'text': data['text'][i],
                'conf': float(data['conf'][i]),
                'bbox': [x, y, x + w, y + h]
            })
    return results

def generate_candidate_phrases(ocr_results, max_gap_x=20, max_diff_y=10):
    """断片的なOCR結果から、連続する単語を結合して候補フレーズのリストを生成する。"""
    candidates = []
    if not ocr_results:
        return []

    # Y座標でソートして、行ごとに処理しやすくする
    ocr_results.sort(key=lambda r: r['bbox'][1])

    for i in range(len(ocr_results)):
        base_word = ocr_results[i]
        
        # 1. 単体の単語を候補として追加
        candidates.append(base_word.copy())
        
        # 2. 後続の単語を結合していく
        combined_text = base_word['text']
        x1, y1, x2, y2 = base_word['bbox']

        # 結合候補を探す
        temp_line = [base_word]
        for j in range(i + 1, len(ocr_results)):
            next_word = ocr_results[j]
            
            # Y座標の中心が近いかチェック
            is_same_line = abs((y1 + y2) / 2 - (next_word['bbox'][1] + next_word['bbox'][3]) / 2) < max_diff_y
            
            if is_same_line:
                temp_line.append(next_word)
        
        # 同じ行内でX座標順にソートし、近接するものを結合
        temp_line.sort(key=lambda r: r['bbox'][0])
        
        current_combined_word = temp_line[0]
        for k in range(1, len(temp_line)):
            next_word_in_line = temp_line[k]
            
            # X座標が近接しているか
            is_adjacent = (next_word_in_line['bbox'][0] - current_combined_word['bbox'][2]) < max_gap_x
            
            if is_adjacent:
                # テキストとBBoxを結合
                current_combined_word['text'] += next_word_in_line['text']
                current_combined_word['bbox'][2] = next_word_in_line['bbox'][2] # 右端を更新
                current_combined_word['bbox'][1] = min(current_combined_word['bbox'][1], next_word_in_line['bbox'][1]) # 上端を更新
                current_combined_word['bbox'][3] = max(current_combined_word['bbox'][3], next_word_in_line['bbox'][3]) # 下端を更新
                current_combined_word['conf'] = min(current_combined_word['conf'], next_word_in_line['conf']) # 信頼度は低い方に合わせる
                
                candidates.append(current_combined_word.copy())

    return candidates


def find_best_match(target_text, candidates):
    """候補リストの中から、ターゲットテキストに最も一致するものを探す。"""
    if not candidates:
        return None

    best_candidate = None
    highest_score = -1.0

    for candidate in candidates:
        candidate_text = candidate['text'].replace(" ", "") # 空白を詰める
        
        # Levenshtein距離を元に類似度を計算
        distance = Levenshtein.distance(target_text, candidate_text)
        longer_len = max(len(target_text), len(candidate_text))
        if longer_len == 0:
            similarity = 1.0 if distance == 0 else 0.0
        else:
            similarity = 1.0 - (distance / longer_len)
        
        # スコアリング: 類似度を主軸にし、文字列長が近いものを優遇
        length_ratio = min(len(target_text), len(candidate_text)) / max(len(target_text), len(candidate_text)) if longer_len > 0 else 0
        score = similarity * 0.8 + length_ratio * 0.2

        if score > highest_score:
            highest_score = score
            best_candidate = candidate
            best_candidate['score'] = score
    
    return best_candidate

def convert_to_global_coords(local_bbox, crop_origin):
    """局所座標をスクリーンショット全体の絶対座標に変換する"""
    local_x1, local_y1, local_x2, local_y2 = local_bbox
    crop_x, crop_y = crop_origin
    return [local_x1 + crop_x, local_y1 + crop_y, local_x2 + crop_x, local_y2 + crop_y]