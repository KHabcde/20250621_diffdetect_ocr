import cv2
import numpy as np
from PIL import Image
import os

def extract_macro_difference_region(
    before_path,
    after_path,
    output_path="macro_diff_region.png",
    threshold=30,          # 差分とみなすピクセルの輝度差しきい値
    min_area=50,           # 無視する最小領域（面積）
    hist_bins=20,          # 差分密度を分析する2Dヒストグラムの分割数
    focus_percentile=90,   # 密集度上位のビンを抽出するためのパーセンタイル
    padding=20             # 最終的に出力する領域の上下左右に加える余白（ピクセル）
):
    """
    2枚の画像から差分を検出し、変化が集中しているマクロ領域を切り出して保存する関数。

    Parameters:
        before_path (str): 差分前の画像パス
        after_path (str): 差分後の画像パス
        output_path (str): 出力画像ファイル名
        threshold (int): 差分の検出しきい値（小さいほど敏感）
        min_area (int): 無視する最小差分領域の面積（小さいほど細かい差分も拾う）
        hist_bins (int): 密度ヒートマップのビン数（小さいほど大きな領域でまとめられる）
        focus_percentile (float): 注目する差分密度の範囲（小さいほど広範囲を拾う）
        padding (int): 最終的に検出された差分領域に追加する余白サイズ（ピクセル単位）

    Returns:
        (x, y, w, h): 差分領域の左上座標とサイズ（幅・高さ） または None
    """

    # -------- 1. 画像の読み込みと形式変換 --------
    before_img = Image.open(before_path)
    after_img = Image.open(after_path)
    before_cv = cv2.cvtColor(np.array(before_img), cv2.COLOR_RGB2BGR)
    after_cv = cv2.cvtColor(np.array(after_img), cv2.COLOR_RGB2BGR)

    # -------- 2. 差分計算 --------
    diff = cv2.absdiff(before_cv, after_cv)  # 各ピクセルの差の絶対値
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)  # グレースケール化
    _, thresh_img = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)  # 2値化（差分強調）

    # -------- 3. 差分領域の抽出 --------
    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h >= min_area:  # 面積フィルタでノイズ除去
            regions.append((x, y, w, h))

    if not regions:
        return None  # 差分がなかった場合

    # -------- 4. 各差分領域の中心座標を取得 --------
    region_centers = np.array([(x + w // 2, y + h // 2) for x, y, w, h in regions])

    # -------- 5. 2Dヒストグラムで密集度を分析 --------
    heatmap, xedges, yedges = np.histogram2d(
        region_centers[:, 0], region_centers[:, 1], bins=hist_bins
    )

    # -------- 6. ヒートマップの上位パーセンタイルを抽出 --------
    flattened = heatmap.flatten()
    threshold_value = np.percentile(flattened[flattened > 0], focus_percentile)
    dense_bins = np.where(heatmap >= threshold_value)

    if dense_bins[0].size == 0:
        return None  # 密集領域がなかった場合

    # -------- 7. 高密度ビンを囲む最小矩形領域を計算 --------
    min_x_bin, max_x_bin = dense_bins[0].min(), dense_bins[0].max()
    min_y_bin, max_y_bin = dense_bins[1].min(), dense_bins[1].max()
    x1, x2 = int(xedges[min_x_bin]), int(xedges[max_x_bin + 1])
    y1, y2 = int(yedges[min_y_bin]), int(yedges[max_y_bin + 1])

    # -------- 8. パディングを追加して領域を拡張 --------
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(after_cv.shape[1], x2 + padding)
    y2 = min(after_cv.shape[0], y2 + padding)

    # -------- 9. 拡張されたマクロ領域を保存 --------
    macro_region = after_cv[y1:y2, x1:x2]
    cv2.imwrite(output_path, macro_region)

    # -------- 10. 結果を返す --------
    return (x1, y1, x2 - x1, y2 - y1)


# ===== 使用例 =====
if __name__ == "__main__":
    before = "before.png"
    after = "after.png"
    output = "macro_diff_region.png"

    coords = extract_macro_difference_region(
        before,
        after,
        output_path=output,
        threshold=30,        # 微小差分検出の閾値：小さくするほど敏感に検出し、マクロ差分領域は広がる方向
        min_area=50,         # 微小差分領域の最小領域：小さくなるほど細かい差分も拾い、マクロ差分領域は広がる方向
        hist_bins=100,       # 微小差分領域の個数集計の際の、画面全体のメッシュ分割数
        focus_percentile=90, # 微小差分領域の個数集計の際に、何パーセント以上が含まれていれば高密度とするか：小さくするほどマクロ差分領域は広がる方向
        padding=50           # 周囲に余白を追加：小さくするほどマクロ差分領域は大きくなる方向
    )

    if coords:
        print(f"マクロ差分領域: 左上=({coords[0]}, {coords[1]}), 幅={coords[2]}, 高さ={coords[3]}")
    else:
        print("差分領域が見つかりませんでした。")
