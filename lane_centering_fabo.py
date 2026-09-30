import cv2
import numpy as np
import sys
import time

# --- Adafruit PCA9685 ライブラリのインポート ---
try:
    import board
    import busio
    from adafruit_pca9685 import PCA9685
    ADAFRUIT_AVAILABLE = True
except ImportError as e:
    print(f"警告: Adafruitライブラリの読み込みに失敗しました ({e})。モックモードで実行します。")
    ADAFRUIT_AVAILABLE = False


# ==========================================
# ★ ステアリング制御用パラメータ設定 ★
# ==========================================
STEER_CHANNEL = 0        # サーボがつながっているPCA9685のチャンネル番号 (0-15)
STEER_CENTER = 325       # ニュートラル（直進）のパルス値
STEER_KP = 0.0           # 位置ズレ（offset）に対する比例ゲイン
STEER_KC = 3600.0         # 曲率（curvature / poly[0]）に対するゲイン（要調整）

# ステアリングの物理的限界
STEER_MIN_PULSE = 150    # 左限界値 (1.0ms 相当)
STEER_MAX_PULSE = 450    # 右限界値 (2.0ms 相当)


# --- 高速化・メモリ再利用用のバッファ ---
_kernel_open = np.ones((3, 3), np.uint8)
_kernel_close = np.ones((7, 3), np.uint8) 
_lpf_kernel = np.ones(5) / 5.0


def init_servo():
    """Adafruit PCA9685 の初期化"""
    if not ADAFRUIT_AVAILABLE:
        return None
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 50  # サーボ用スタンダードPWM周波数 (50Hz)
        time.sleep(0.1)
        
        center_duty = int(STEER_CENTER * 65535 / 4096)
        pca.channels[STEER_CHANNEL].duty_cycle = center_duty
        print("Adafruit PCA9685 の初期化に成功しました。")
        return pca
    except Exception as e:
        print(f"エラー: PCA9685の初期化に失敗しました -> {e}")
        return None


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1280,
    capture_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True max-buffers=1 sync=False"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            capture_width,
            capture_height,
        )
    )


def calculate_steering_offset(frame):
    h, w, _ = frame.shape
    x_center_frame = w // 2
    
    # 1. ROI（解析領域）の設定
    roi_top = int(h * 0.5)
    roi_bottom = int(h * 0.99)
    roi = frame[roi_top:roi_bottom, :]
    
    proc_w = 320
    scale = proc_w / w
    proc_h = int(roi.shape[0] * scale)
    small_roi = cv2.resize(roi, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
    
    hsv_roi = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV)
    
    # 2. 床色の動的サンプリング（手前中央付近）
    sample_y_start = int(proc_h * 0.80)
    sample_x_start = int(proc_w * 0.35)
    sample_x_end = int(proc_w * 0.65)

    ground_patch = hsv_roi[sample_y_start:proc_h, sample_x_start:sample_x_end]
    mean_h, mean_s, mean_v = cv2.mean(ground_patch)[:3]
    
    # 明度(V)・彩度(S)の範囲調整
    min_v_allowed = max(0, int(mean_v - 70))
    max_v_allowed = min(220, int(mean_v + 50))  # 白い壁を除外するため上限を少し抑える
    max_s_allowed = max(60, int(mean_s + 40))

    lower_ground = np.array([0, 0, min_v_allowed], dtype=np.uint8)
    upper_ground = np.array([180, max_s_allowed, max_v_allowed], dtype=np.uint8)

    mask_ground = cv2.inRange(hsv_roi, lower_ground, upper_ground)
    
    # ノイズ除去と穴埋め
    cv2.morphologyEx(mask_ground, cv2.MORPH_OPEN, _kernel_open, dst=mask_ground)
    cv2.morphologyEx(mask_ground, cv2.MORPH_CLOSE, _kernel_close, dst=mask_ground)
    
    # 3. 重心ベース ＆ 壁からの反発バイアス付きスキャン
    max_rows = proc_h
    y_pts_buf = np.empty(max_rows, dtype=np.float32)
    x_pts_buf = np.empty(max_rows, dtype=np.float32)
    valid_count = 0
    
    # ★ 左右の境界（壁）座標保持用リスト
    left_wall_pts = []
    right_wall_pts = []

    min_road_width_px = 45
    max_x_jump_px = 25
    consecutive_limit = 15  # 奥まで追従できるよう上限を拡大
    missing_count = 0
    y_cutoff_real = roi_top
    
    last_x = None

    for y in range(proc_h - 1, -1, -1):
        if y < int(proc_h * 0.35):
            y_cutoff_real = int(y / scale) + roi_top
            break

        x_indices = np.where(mask_ground[y, :] > 0)[0]
        
        if len(x_indices) > 0:
            splits = np.where(np.diff(x_indices) > 1)[0] + 1
            clusters = np.split(x_indices, splits)
            
            if last_x is None:
                best_cluster = max(clusters, key=len)
            else:
                valid_clusters = [c for c in clusters if abs(np.mean(c) - last_x) <= max_x_jump_px]
                if valid_clusters:
                    best_cluster = max(valid_clusters, key=len)
                else:
                    best_cluster = []

            if len(best_cluster) >= min_road_width_px:
                left_edge = best_cluster[0]
                right_edge = best_cluster[-1]

                # ★ 画面座標系に変換して左右の端点を保存
                real_y = int(y / scale) + roi_top
                real_left_x = int(left_edge / scale)
                real_right_x = int(right_edge / scale)

                left_wall_pts.append([real_left_x, real_y])
                right_wall_pts.append([real_right_x, real_y])

                x_mean = np.mean(best_cluster)
                
                # 壁反発処理
                wall_margin_thresh = 60
                if left_edge < wall_margin_thresh:
                    x_mean += (wall_margin_thresh - left_edge) * 0.8
                elif (proc_w - right_edge) < wall_margin_thresh:
                    x_mean -= (wall_margin_thresh - (proc_w - right_edge)) * 0.8

                last_x = x_mean
                missing_count = 0
                
                y_pts_buf[valid_count] = real_y
                x_pts_buf[valid_count] = int(x_mean / scale)
                valid_count += 1
            else:
                missing_count += 1
        else:
            missing_count += 1
            
        if missing_count >= consecutive_limit:
            y_cutoff_real = int(y / scale) + roi_top
            break
            
    output = frame.copy()
    
    # サンプリング領域の描画
    dbg_sample_y1 = int(sample_y_start / scale) + roi_top
    dbg_sample_y2 = int(proc_h / scale) + roi_top
    dbg_sample_x1 = int(sample_x_start / scale)
    dbg_sample_x2 = int(sample_x_end / scale)
    cv2.rectangle(output, (dbg_sample_x1, dbg_sample_y1), (dbg_sample_x2, dbg_sample_y2), (0, 255, 255), 2)
    cv2.putText(output, "Sampling Area", (dbg_sample_x1, dbg_sample_y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # 緑色オーバーレイ表示
    mask_ground_full = cv2.resize(mask_ground, (w, roi_bottom - roi_top), interpolation=cv2.INTER_NEAREST)
    roi_sub = output[roi_top:roi_bottom, :]
    green_cond = mask_ground_full > 0
    roi_sub[green_cond] = (roi_sub[green_cond] * 0.7 + np.array([0, 255, 0], dtype=np.float32) * 0.3).astype(np.uint8)

    # ★ 左右の境界（壁）を赤色の帯としてハイライト表示
    if len(left_wall_pts) > 1:
        pts_left = np.array(left_wall_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [pts_left], isClosed=False, color=(0, 0, 255), thickness=6)

    if len(right_wall_pts) > 1:
        pts_right = np.array(right_wall_pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(output, [pts_right], isClosed=False, color=(0, 0, 255), thickness=6)

    # 4. LPF + 重み付き2次関数フィッティング
    if valid_count >= 5:
        y_pts = y_pts_buf[:valid_count]
        x_pts = x_pts_buf[:valid_count]
        x_lpf = np.convolve(x_pts, _lpf_kernel, mode='same')
        
        dev = np.abs(x_pts - x_lpf)
        stability_weights = np.exp(-0.5 * (dev / 15.0) ** 2)
        y_norm = (y_pts - y_cutoff_real) / max(1, (roi_bottom - y_cutoff_real))
        final_weights = ((y_norm ** 2) + 0.05) * stability_weights
        
        # アンカーポイントの設定
        anchor_y = np.array([roi_bottom, roi_bottom - 4, roi_bottom - 8], dtype=np.float32)
        anchor_x = np.array([x_center_frame, x_center_frame, x_center_frame], dtype=np.float32)
        anchor_weights = np.array([200.0, 100.0, 50.0], dtype=np.float32)
        
        y_pts_fixed = np.concatenate([y_pts, anchor_y])
        x_lpf_fixed = np.concatenate([x_lpf, anchor_x])
        final_weights_fixed = np.concatenate([final_weights, anchor_weights])
        
        poly = np.polyfit(y_pts_fixed - roi_bottom, x_lpf_fixed, 2, w=final_weights_fixed)
        
        plot_y = np.linspace(roi_bottom, y_cutoff_real, 30)
        plot_x = np.polyval(poly, plot_y - roi_bottom)
        
        curve_pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
        cv2.polylines(output, [curve_pts], isClosed=False, color=(0, 255, 255), thickness=4)
        
        target_x_bot = int(np.polyval(poly, 0))
        offset = target_x_bot - x_center_frame
        cv2.circle(output, (target_x_bot, roi_bottom - 10), 8, (0, 255, 255), -1)
        
        curvature = poly[0]
        slope = poly[1]
    else:
        # ★ 点数が不足している場合は画面中央に真っ直ぐの直線を描画
        offset = 0
        curvature = 0.0
        slope = 0.0

        straight_pts = np.array([
            [x_center_frame, roi_bottom],
            [x_center_frame, roi_top]
        ], dtype=np.int32)

        cv2.polylines(output, [straight_pts], isClosed=False, color=(0, 255, 255), thickness=4)
        cv2.circle(output, (x_center_frame, roi_bottom - 10), 8, (0, 255, 255), -1)

    cv2.line(output, (x_center_frame, roi_top), (x_center_frame, roi_bottom), (255, 0, 0), 1)
    return output, offset, curvature, slope


def drive_control(offset, curvature, pca=None):
    pulse_change = (STEER_KP * offset) + (STEER_KC * curvature)
    target_pulse = int(STEER_CENTER + pulse_change)
    target_pulse = max(STEER_MIN_PULSE, min(STEER_MAX_PULSE, target_pulse))

    if pca is not None:
        duty = int(target_pulse * 65535 / 4096)
        pca.channels[STEER_CHANNEL].duty_cycle = duty

    print(f"\rOffset: {offset:+4d} px | Curv: {curvature:+.5f} | Target Pulse: {target_pulse:4d}", end="", flush=True)


def main():
    pca = init_servo()

    gst_str = gstreamer_pipeline()
    cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("エラー: カメラを開けませんでした。")
        sys.exit(1)

    print("リアルタイム追従を開始します。'q' キーで終了します。")

    TARGET_DISPLAY_FPS = 5
    display_interval = 1.0 / TARGET_DISPLAY_FPS
    last_display_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nフレームの取得に失敗しました。")
                break

            output_frame, offset, curvature, slope = calculate_steering_offset(frame)
            drive_control(offset, curvature, pca)
            
            current_time = time.time()
            if current_time - last_display_time >= display_interval:
                cv2.imshow("Lane Centering", cv2.resize(output_frame, (480, 270)))
                last_display_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n停止シグナルを受信しました。")
                break

    finally:
        if pca is not None:
            print("\nステアリングをセンターに復帰しています...")
            pca.channels[STEER_CHANNEL].duty_cycle = int(STEER_CENTER * 65535 / 4096)
            time.sleep(0.2)

        cap.release()
        cv2.destroyAllWindows()
        print("正常にリソースを解放して終了しました。")


if __name__ == "__main__":
    main()