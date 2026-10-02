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
# ★ 認識領域 & ステアリング制御用パラメータ設定 ★
# ==========================================
ROI_TOP_RATIO = 0.40       # 上端位置
ROI_BOTTOM_RATIO = 0.99    # 下端位置
SCAN_LIMIT_RATIO = 0.25

# トラックバーで調整した最適値
WHITE_V_MIN = 170
GROUND_H_TOL = 90
GROUND_S_TOL = 40
GROUND_V_TOL = 70
WHITE_V_MIN = 170
GROUND_H_TOL = 90
GROUND_S_TOL = 40
GROUND_V_TOL = 60

STEER_CHANNEL = 0
STEER_CENTER = 325

# 制御ゲイン
STEER_KP = 0.0             # 重心オフセットPゲイン (px -> pulse)
STEER_KP = 0.8             # 重心オフセットPゲイン (px -> pulse)
# STEER_KP = 0.8             # 重心オフセットPゲイン (px -> pulse)
# STEER_KC = 3600.0          # 曲率ゲイン (curvature -> pulse)
# STEER_KC = 5400.0          # 曲率ゲイン (curvature -> pulse)
STEER_KC = 100000.0          # 曲率ゲイン (curvature -> pulse)
WALL_REPULSION_GAIN = 0.0   # ★ 壁からの反発力ゲインを強化 (1.5 -> 3.0)
WALL_REPULSION_GAIN = 20.0   # ★ 壁からの反発力ゲインを強化 (1.5 -> 3.0)
WALL_REPULSION_GAIN = 2.0   # ★ 壁からの反発力ゲインを強化 (1.5 -> 3.0)

STEER_MIN_PULSE = 150
STEER_MAX_PULSE = 450


# --- 高速化・メモリ再利用用のバッファ ---
_kernel_open = np.ones((3, 3), np.uint8)
_kernel_close = np.ones((7, 3), np.uint8) 
_lpf_kernel = np.ones(5) / 5.0


def init_servo():
    if not ADAFRUIT_AVAILABLE:
        return None
    try:
        i2c = busio.I2C(board.SCL, board.SDA)
        pca = PCA9685(i2c)
        pca.frequency = 50
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


def remove_isolated_wall_noise(mask_wall, min_area=80):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_wall, connectivity=8)
    filtered_mask = np.zeros_like(mask_wall)
    area_idx = getattr(cv2, 'CC_STAT_AREA', getattr(cv2, 'STAT_AREA', 4))
    
    for i in range(1, num_labels):
        area = stats[i, area_idx]
        if area >= min_area:
            filtered_mask[labels == i] = 255
            
    return filtered_mask


def calculate_wall_repulsion(mask_wall):
    proc_h, proc_w = mask_wall.shape
    near_h_start = 0
    near_wall = mask_wall[near_h_start:, :]
    
    mid_x = proc_w // 2
    left_wall = near_wall[:, :mid_x]
    right_wall = near_wall[:, mid_x:]
    
    y_weights = np.linspace(0.2, 1.0, near_wall.shape[0])[:, None]
    
    left_score = np.sum((left_wall > 0) * y_weights)
    right_score = np.sum((right_wall > 0) * y_weights)
    
    max_score = (near_wall.shape[0] * near_wall.shape[1] / 2.0) * 0.6
    
    repulsion_force = (left_score - right_score) / max(1.0, max_score)
    repulsion_force = np.clip(repulsion_force, -1.0, 1.0)
    
    repulsion_offset_px = repulsion_force * 100.0 * WALL_REPULSION_GAIN
    return repulsion_offset_px


def calculate_steering_offset(frame):
    h, w, _ = frame.shape
    x_center_frame = w // 2

    # 1. ROI設定
    roi_top = int(h * ROI_TOP_RATIO)
    roi_bottom = int(h * ROI_BOTTOM_RATIO)
    roi = frame[roi_top:roi_bottom, :]
    
    proc_w = 320
    scale = proc_w / w
    proc_h = int(roi.shape[0] * scale)
    small_roi = cv2.resize(roi, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
    
    hsv_roi = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV)
    
    # 2. 床色の動的サンプリング
    sample_y_start = int(proc_h * 0.85)
    sample_x_start = int(proc_w * 0.45)
    sample_x_end = int(proc_w * 0.55)

    ground_patch = hsv_roi[sample_y_start:proc_h, sample_x_start:sample_x_end]
    mean_h, mean_s, mean_v = cv2.mean(ground_patch)[:3]
    
    lower_ground = np.array([max(0, int(mean_h - GROUND_H_TOL)), max(0, int(mean_s - GROUND_S_TOL)), max(0, int(mean_v - GROUND_V_TOL))], dtype=np.uint8)
    upper_ground = np.array([min(180, int(mean_h + GROUND_H_TOL)), min(255, int(mean_s + GROUND_S_TOL)), min(255, int(mean_v + GROUND_V_TOL))], dtype=np.uint8)
    mask_ground = cv2.inRange(hsv_roi, lower_ground, upper_ground)

    # 壁の検出（白・赤）
    lower_white = np.array([0, 0, WHITE_V_MIN], dtype=np.uint8)
    upper_white = np.array([180, 50, 255], dtype=np.uint8)
    mask_white = cv2.inRange(hsv_roi, lower_white, upper_white)

    lower_red1 = np.array([0, 100, 80], dtype=np.uint8)
    upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([170, 100, 80], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask_red1 = cv2.inRange(hsv_roi, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_roi, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_wall_raw = cv2.bitwise_or(mask_white, mask_red)
    mask_wall = cv2.bitwise_and(mask_wall_raw, cv2.bitwise_not(mask_ground))
    cv2.morphologyEx(mask_wall, cv2.MORPH_CLOSE, _kernel_close, dst=mask_wall)
    mask_wall = remove_isolated_wall_noise(mask_wall, min_area=80)

    repulsion_offset = calculate_wall_repulsion(mask_wall)

    cv2.morphologyEx(mask_ground, cv2.MORPH_OPEN, _kernel_open, dst=mask_ground)
    cv2.morphologyEx(mask_ground, cv2.MORPH_CLOSE, _kernel_close, dst=mask_ground)

    # 3. 壁ベースの床領域スキャン
    max_rows = proc_h
    y_pts_buf = np.empty(max_rows, dtype=np.float32)
    x_pts_buf = np.empty(max_rows, dtype=np.float32)
    valid_count = 0

    scan_limit_y = int(proc_h * SCAN_LIMIT_RATIO)
    y_cutoff_real = roi_top
    
    last_left_edge = None
    last_right_edge = None

    for y in range(proc_h - 1, scan_limit_y - 1, -1):
        wall_x_indices = np.where(mask_wall[y, :] > 0)[0]
        
        left_wall_candidates = wall_x_indices[wall_x_indices < proc_w // 2]
        right_wall_candidates = wall_x_indices[wall_x_indices >= proc_w // 2]
        
        current_left = left_wall_candidates[-1] if len(left_wall_candidates) > 0 else None
        current_right = right_wall_candidates[0] if len(right_wall_candidates) > 0 else None
        
        if current_left is None and last_left_edge is not None:
            current_left = last_left_edge
        if current_right is None and last_right_edge is not None:
            current_right = last_right_edge

        if current_left is not None and current_right is not None and current_left < current_right:
            road_pixels = np.arange(current_left, current_right + 1)
            x_mean = np.mean(road_pixels)
            
            last_left_edge = current_left
            last_right_edge = current_right
            
            real_y = int(y / scale) + roi_top
            y_pts_buf[valid_count] = real_y
            x_pts_buf[valid_count] = int(x_mean / scale)
            valid_count += 1
        else:
            ground_indices = np.where(mask_ground[y, :] > 0)[0]
            if len(ground_indices) > 10:
                x_mean = np.mean(ground_indices)
                real_y = int(y / scale) + roi_top
                y_pts_buf[valid_count] = real_y
                x_pts_buf[valid_count] = int(x_mean / scale)
                valid_count += 1
            else:
                y_cutoff_real = int(y / scale) + roi_top
                break
            
    output = frame.copy()

    mask_ground_full = cv2.resize(mask_ground, (w, roi_bottom - roi_top), interpolation=cv2.INTER_NEAREST)
    mask_wall_full = cv2.resize(mask_wall, (w, roi_bottom - roi_top), interpolation=cv2.INTER_NEAREST)
    
    roi_sub = output[roi_top:roi_bottom, :]
    
    green_cond = mask_ground_full > 0
    roi_sub[green_cond] = (roi_sub[green_cond] * 0.7 + np.array([0, 255, 0], dtype=np.float32) * 0.3).astype(np.uint8)

    red_cond = mask_wall_full > 0
    roi_sub[red_cond] = (roi_sub[red_cond] * 0.6 + np.array([0, 0, 255], dtype=np.float32) * 0.4).astype(np.uint8)

    dbg_sample_y1 = int(sample_y_start / scale) + roi_top
    dbg_sample_y2 = int(proc_h / scale) + roi_top
    dbg_sample_x1 = int(sample_x_start / scale)
    dbg_sample_x2 = int(sample_x_end / scale)
    cv2.rectangle(output, (dbg_sample_x1, dbg_sample_y1), (dbg_sample_x2, dbg_sample_y2), (0, 255, 255), 2)

    cv2.rectangle(output, (0, roi_top), (w, roi_bottom), (255, 0, 0), 2)
    cv2.putText(output, "Repulsion Area (Full ROI)", (10, roi_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # 4. 曲線/直線フィッティングと描画
    center_offset = 0.0
    curvature = 0.0
    slope = 0.0
    fit_status_str = "Initializing..."

    draw_bottom_y = min(h - 5, roi_bottom + int((h - roi_bottom) * 0.5))

    if valid_count > 0:
        for i in range(valid_count):
            cv2.circle(output, (int(x_pts_buf[i]), int(y_pts_buf[i])), 4, (255, 0, 255), -1)

    if valid_count >= 3:
        y_pts = y_pts_buf[:valid_count]
        x_pts = x_pts_buf[:valid_count]
        
        if len(x_pts) >= 5:
            x_lpf = np.convolve(x_pts, _lpf_kernel, mode='same')
        else:
            x_lpf = x_pts.copy()
        
        dev = np.abs(x_pts - x_lpf)
        stability_weights = np.exp(-0.5 * (dev / 15.0) ** 2)
        y_norm = (y_pts - y_cutoff_real) / max(1, (roi_bottom - y_cutoff_real))
        final_weights = ((y_norm ** 2) + 0.05) * stability_weights
        
        anchor_y = np.array([roi_bottom, roi_bottom - 4, roi_bottom - 8], dtype=np.float32)
        anchor_x = np.array([x_center_frame, x_center_frame, x_center_frame], dtype=np.float32)
        anchor_weights = np.array([200.0, 100.0, 50.0], dtype=np.float32)
        
        y_pts_fixed = np.concatenate([y_pts, anchor_y])
        x_lpf_fixed = np.concatenate([x_lpf, anchor_x])
        final_weights_fixed = np.concatenate([final_weights, anchor_weights])
        
        try:
            poly = np.polyfit(y_pts_fixed - roi_bottom, x_lpf_fixed, 2, w=final_weights_fixed)
            
            plot_y = np.linspace(draw_bottom_y, y_cutoff_real, 40)
            plot_x = np.polyval(poly, plot_y - roi_bottom)
            
            curve_pts = np.column_stack((plot_x, plot_y)).astype(np.int32)
            cv2.polylines(output, [curve_pts], isClosed=False, color=(0, 255, 255), thickness=4)
            
            target_x_bot = int(np.polyval(poly, 0))
            center_offset = target_x_bot - x_center_frame
            
            curvature = poly[0]
            slope = poly[1]
            fit_status_str = "Polyfit OK (Quadratic)"
        except Exception as e:
            mean_x = int(np.mean(x_pts))
            cv2.line(output, (mean_x, draw_bottom_y), (mean_x, y_cutoff_real), (0, 255, 255), 4)
            center_offset = mean_x - x_center_frame
            fit_status_str = f"Polyfit Exception: {type(e).__name__}"
    else:
        cv2.line(output, (x_center_frame, draw_bottom_y), (x_center_frame, roi_top), (0, 255, 255), 2)
        fit_status_str = "Too Few Points (< 3)"

    final_offset = center_offset + repulsion_offset
    final_target_x = int(x_center_frame + final_offset)
    
    cv2.circle(output, (final_target_x, draw_bottom_y - 15), 10, (255, 0, 255), -1)
    cv2.line(output, (x_center_frame, roi_top), (x_center_frame, draw_bottom_y), (255, 0, 0), 1)

    p_pulse_contrib = STEER_KP * final_offset
    c_pulse_contrib = STEER_KC * curvature
    total_pulse_change = p_pulse_contrib + c_pulse_contrib
    raw_pulse = STEER_CENTER + total_pulse_change
    target_pulse = int(np.clip(raw_pulse, STEER_MIN_PULSE, STEER_MAX_PULSE))

    # 詳細デバッグ情報のオーバーレイ表示エリア
    overlay_bg = output[10:200, 10:520].copy()
    cv2.rectangle(output, (10, 10), (520, 200), (0, 0, 0), -1)
    cv2.addWeighted(output[10:200, 10:520], 0.6, overlay_bg, 0.4, 0, output[10:200, 10:520])

    cv2.putText(output, f"Valid Points   : {valid_count} (Req >= 3)", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(output, f"Fit Status     : {fit_status_str}", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 255, 200), 1)
    cv2.putText(output, f"P-Term (Offset): {final_offset:+6.1f}px -> {p_pulse_contrib:+6.1f} us", 
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(output, f"  |- Center Offset : {center_offset:+6.1f}px", 
                (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(output, f"  |- Repulsion     : {repulsion_offset:+6.1f}px", 
                (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 100, 255), 1)
    cv2.putText(output, f"C-Term (Curv)    : {curvature:+8.5f} -> {c_pulse_contrib:+6.1f} us", 
                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(output, f"Final Pulse      : {target_pulse} us", 
                (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    plain_view = frame.copy()
    cv2.putText(plain_view, "Original (No Overlay)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    combined_view = np.hstack((plain_view, output))

    return combined_view, final_offset, curvature, target_pulse


def drive_control(target_pulse, pca=None):
    if pca is not None:
        duty = int(target_pulse * 65535 / 4096)
        pca.channels[STEER_CHANNEL].duty_cycle = duty

    print(f"\rTarget Pulse: {target_pulse:4d}", end="", flush=True)


def main():
    pca = init_servo()

    gstr = gstreamer_pipeline()
    cap = cv2.VideoCapture(gstr, cv2.CAP_GSTREAMER)

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

            combined_frame, offset, curvature, target_pulse = calculate_steering_offset(frame)
            drive_control(target_pulse, pca)
            
            current_time = time.time()
            if current_time - last_display_time >= display_interval:
                cv2.imshow("Lane Centering (Left: Original | Right: Processed)", cv2.resize(combined_frame, (1280, 360)))
                last_display_time = current_time

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n停止シグナルを受信しました。")
                break

    finally:
        if pca is not None:
            print("\nステアリングをセンターに復帰しています...")
            pca.channels[STEER_CHANNEL].duty_cycle = int(STEER_CENTER * 65535 / 4096)
            time.sleep(0.2)

        cap.reset() if hasattr(cap, 'reset') else None
        cap.release()
        cv2.destroyAllWindows()
        print("正常にリソースを解放して終了しました。")


if __name__ == "__main__":
    main()