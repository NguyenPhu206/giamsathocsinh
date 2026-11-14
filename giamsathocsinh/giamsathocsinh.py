import cv2
import os
import glob
import math
import time
import requests
import gradio as gr
import yt_dlp
from ultralytics import YOLO

# ==============================================================================
# 1. KHỞI TẠO MÔ HÌNH AI (TỐI ƯU CHO HP ELITEBOOK 830 G5)
# ==============================================================================
print("⏳ Đang khởi tạo mô hình AI nhận diện 4 hành vi...")
model_obj = YOLO("yolov8n.pt")       # Nhận diện điện thoại (class 67)
model_pose = YOLO("yolov8n-pose.pt") # Nhận diện 17 điểm khớp xương cơ thể

CUSTOM_MODEL_PATH = "best.pt"
model_custom = YOLO(CUSTOM_MODEL_PATH) if os.path.exists(CUSTOM_MODEL_PATH) else None
print("✅ Mô hình AI đã sẵn sàng!")


# ==============================================================================
# 2. HÀM TẢI VIDEO TỪ ĐƯỜNG LINK (TIKTOK, YOUTUBE, FACEBOOK...)
# ==============================================================================
def tai_video_tu_link(url):
    """Tải video từ link TikTok (qua API TikWM) và YouTube/Facebook (qua yt-dlp)"""
    url = url.strip()
    output_filename = "video_tu_link.mp4"

    for f in glob.glob("video_tu_link.*"):
        try:
            os.remove(f)
        except Exception:
            pass

    # 1. Xử lý link TikTok
    if "tiktok.com" in url.lower():
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        try:
            api_url = "https://www.tikwm.com/api/"
            resp = requests.post(api_url, data={"url": url, "hd": 1}, headers=headers, timeout=15).json()
            if resp.get("code") == 0 and "data" in resp:
                mp4_link = resp["data"].get("play") or resp["data"].get("wmplay")
                if mp4_link:
                    if mp4_link.startswith("/"):
                        mp4_link = "https://www.tikwm.com" + mp4_link
                    video_bytes = requests.get(mp4_link, headers=headers, timeout=30).content
                    with open(output_filename, "wb") as f:
                        f.write(video_bytes)
                    if os.path.exists(output_filename) and os.path.getsize(output_filename) > 10000:
                        return output_filename
        except Exception as e:
            print("Chuyển sang phương án dự phòng cho TikTok:", e)

    # 2. Xử lý link YouTube và các nền tảng khác
    if "youtube.com" in url.lower() or "youtu.be" in url.lower():
        url = url.replace("m.youtube.com", "www.youtube.com")
        if "&" in url and "watch?v=" in url:
            url = url.split("&")[0]

    ydl_opts = {
        'format': 'b/best',
        'outtmpl': 'video_tu_link.%(ext)s',
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'extractor_args': {'youtube': {'player_client': ['ios', 'android', 'web']}}
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception:
        ydl_opts_fallback = {
            'outtmpl': 'video_tu_link.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts_fallback) as ydl:
            ydl.download([url])

    cac_file_tai_ve = glob.glob("video_tu_link.*")
    if len(cac_file_tai_ve) > 0:
        return cac_file_tai_ve[0]
    return None


# ==============================================================================
# 3. CÁC HÀM TOÁN HỌC PHÂN TÍCH HÀNH VI HỌC SINH
# ==============================================================================
def tinh_khoang_cach(p1, p2):
    """Tính khoảng cách Euclid giữa 2 điểm (x1, y1) và (x2, y2)"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def kiem_tra_dien_thoai_gan_nguoi(phone_boxes, person_box, keypoints):
    """HÀNH VI 1: Kiểm tra điện thoại nằm trong vùng cơ thể hoặc gần cổ tay học sinh"""
    px1, py1, px2, py2 = person_box
    left_wrist = keypoints[9] if len(keypoints) > 9 else [0, 0]
    right_wrist = keypoints[10] if len(keypoints) > 10 else [0, 0]

    for (bx1, by1, bx2, by2, conf) in phone_boxes:
        phone_cx = (bx1 + bx2) / 2
        phone_cy = (by1 + by2) / 2

        trong_khung_nguoi = (px1 - 25 <= phone_cx <= px2 + 25) and (py1 - 25 <= phone_cy <= py2 + 45)
        gan_co_tay_trai = (left_wrist[0] > 0 and tinh_khoang_cach((phone_cx, phone_cy), left_wrist) < 140)
        gan_co_tay_phai = (right_wrist[0] > 0 and tinh_khoang_cach((phone_cx, phone_cy), right_wrist) < 140)

        if trong_khung_nguoi or gan_co_tay_trai or gan_co_tay_phai:
            return True, (bx1, by1, bx2, by2, conf)

    return False, None


def phan_tich_guc_dau_xuong_ban(keypoints, nguong_guc_dau):
    """
    HÀNH VI 2: GỤC ĐẦU XUỐNG BÀN (Chặt chẽ hơn, tránh nhầm với cúi viết bài nhẹ)
    Kích hoạt khi:
    - Tỷ lệ khoảng cách dọc từ Đầu đến Vai nhỏ hơn ngưỡng (đầu hạ sát mặt bàn).
    - Hoặc học sinh úp mặt vào cánh tay trên bàn (mũi bị che khuất nhưng 2 tai hạ thấp sát vai).
    """
    if len(keypoints) < 7:
        return False

    nose = keypoints[0]
    l_eye, r_eye = keypoints[1], keypoints[2]
    l_ear, r_ear = keypoints[3], keypoints[4]
    l_sh, r_sh = keypoints[5], keypoints[6]

    if l_sh[0] <= 0 or r_sh[0] <= 0:
        return False

    sh_mid_y = (l_sh[1] + r_sh[1]) / 2
    shoulder_width = tinh_khoang_cach(l_sh, r_sh)
    if shoulder_width < 25:
        return False

    # Xác định tọa độ Y của phần đầu
    if nose[1] > 0:
        head_y = nose[1]
    elif l_eye[1] > 0 and r_eye[1] > 0:
        head_y = (l_eye[1] + r_eye[1]) / 2
    elif l_ear[1] > 0 and r_ear[1] > 0:
        head_y = (l_ear[1] + r_ear[1]) / 2
    elif l_ear[1] > 0 or r_ear[1] > 0:
        head_y = l_ear[1] if l_ear[1] > 0 else r_ear[1]
    else:
        # Nếu toàn bộ mũi, mắt, tai đều bị che khuất do úp sấp mặt xuống bàn mà vẫn thấy 2 vai
        return True

    # Tỷ lệ chiều cao từ trung điểm hai vai lên đầu chia cho độ rộng vai
    head_ratio = (sh_mid_y - head_y) / shoulder_width

    # Khi gục đầu xuống bàn, đầu hạ xuống rất gần hoặc thấp hơn đường nối hai vai
    if head_ratio < nguong_guc_dau:
        return True

    return False


def phan_tich_gio_tay_phat_bieu(keypoints):
    """
    HÀNH VI 3: GIƠ TAY PHÁT BIỂU
    Kích hoạt khi Cổ tay (Trái hoặc Phải) đưa lên cao vượt qua đường ngang Vai,
    đồng thời không bị trùng với tư thế chống cằm (cổ tay phải cao hơn mũi hoặc cách xa mũi).
    """
    if len(keypoints) < 11:
        return False

    nose = keypoints[0]
    l_sh, r_sh = keypoints[5], keypoints[6]
    l_el, r_el = keypoints[7], keypoints[8]
    l_wr, r_wr = keypoints[9], keypoints[10]

    if l_sh[0] <= 0 or r_sh[0] <= 0:
        return False

    shoulder_width = tinh_khoang_cach(l_sh, r_sh)
    if shoulder_width < 25:
        return False

    # Kiểm tra tay trái giơ lên (Trong hệ tọa độ ảnh, càng lên cao giá trị Y càng nhỏ)
    tay_trai_gio = False
    if l_wr[1] > 0:
        cao_hon_vai = (l_sh[1] - l_wr[1]) > (0.25 * shoulder_width)
        cao_hon_mui = (nose[1] > 0 and l_wr[1] < nose[1] + 0.15 * shoulder_width)
        khong_phai_chong_cam = True
        if nose[0] > 0 and tinh_khoang_cach(l_wr, nose) < (0.35 * shoulder_width) and l_wr[1] >= nose[1]:
            khong_phai_chong_cam = False

        if (cao_hon_vai or cao_hon_mui) and khong_phai_chong_cam:
            if l_el[1] <= 0 or l_wr[1] < l_el[1] + 15:
                tay_trai_gio = True

    # Kiểm tra tay phải giơ lên
    tay_phai_gio = False
    if r_wr[1] > 0:
        cao_hon_vai = (r_sh[1] - r_wr[1]) > (0.25 * shoulder_width)
        cao_hon_mui = (nose[1] > 0 and r_wr[1] < nose[1] + 0.15 * shoulder_width)
        khong_phai_chong_cam = True
        if nose[0] > 0 and tinh_khoang_cach(r_wr, nose) < (0.35 * shoulder_width) and r_wr[1] >= nose[1]:
            khong_phai_chong_cam = False

        if (cao_hon_vai or cao_hon_mui) and khong_phai_chong_cam:
            if r_el[1] <= 0 or r_wr[1] < r_el[1] + 15:
                tay_phai_gio = True

    return tay_trai_gio or tay_phai_gio


# ==============================================================================
# 4. HÀM SUY LUẬN AI & HÀM VẼ ĐỘC LẬP (GIÚP CAMERA REAL-TIME MƯỢT 30 FPS)
# ==============================================================================
def suy_luan_ai_tren_frame(frame, current_time, trackers_state, conf_phone, nguong_guc_dau, tg_guc_dau, tg_dien_thoai):
    """
    Chạy YOLO để lấy danh sách tọa độ vẽ (overlay_data) và danh sách cảnh báo.
    Tách rời bước tính toán AI và bước vẽ giúp tái sử dụng kết quả cho các frame xen kẽ.
    """
    ds_canh_bao = []
    ds_tich_cuc = []
    overlay_items = []

    # 1. Quét tìm điện thoại (imgsz=384 giúp CPU Intel UHD 620 chạy cực nhanh)
    results_obj = model_obj.predict(frame, imgsz=384, classes=[67], conf=conf_phone, verbose=False)
    phone_boxes = []
    for box in results_obj[0].boxes:
        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
        phone_boxes.append((bx1, by1, bx2, by2, float(box.conf[0])))

    if model_custom is not None:
        res_custom = model_custom.predict(frame, imgsz=384, conf=conf_phone, verbose=False)
        for box in res_custom[0].boxes:
            cls_name = model_custom.names[int(box.cls[0])].lower()
            if "phone" in cls_name or "dien_thoai" in cls_name:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                phone_boxes.append((bx1, by1, bx2, by2, float(box.conf[0])))

    # 2. Theo vết từng học sinh và trích xuất khớp xương
    results_pose = model_pose.track(frame, imgsz=384, conf=0.45, persist=True, tracker="bytetrack.yaml", verbose=False)

    if results_pose[0].boxes is not None and results_pose[0].keypoints is not None:
        boxes = results_pose[0].boxes.xyxy.cpu().numpy()
        keypoints_all = results_pose[0].keypoints.xy.cpu().numpy()

        if results_pose[0].boxes.id is not None:
            track_ids = results_pose[0].boxes.id.int().cpu().tolist()
        else:
            track_ids = list(range(1, len(boxes) + 1))

        for idx, track_id in enumerate(track_ids):
            person_box = list(map(int, boxes[idx]))
            kpts = keypoints_all[idx]

            if track_id not in trackers_state:
                trackers_state[track_id] = {
                    "guc_start": None,
                    "phone_start": None,
                    "logged_guc": False,
                    "logged_phone": False,
                    "logged_giotay": False
                }
            st = trackers_state[track_id]

            # --- KIỂM TRA 3 HÀNH VI ĐẶC BIỆT (NẾU KHÔNG CÓ THÌ LÀ NGỒI HỌC NGHIÊM TÚC) ---
            # A. Hành vi Gục đầu xuống bàn
            is_guc_dau = phan_tich_guc_dau_xuong_ban(kpts, nguong_guc_dau)
            vi_pham_guc = False
            guc_dur = 0.0
            if is_guc_dau:
                if st["guc_start"] is None:
                    st["guc_start"] = current_time
                guc_dur = current_time - st["guc_start"]
                if guc_dur >= tg_guc_dau:
                    vi_pham_guc = True
                    ds_canh_bao.append(f"🔴 HS #{track_id}: GỤC ĐẦU XUỐNG BÀN ({guc_dur:.1f}s)")
                    if not st["logged_guc"]:
                        trackers_state["history_log"].append(f"[{current_time:.1f}s] HS #{track_id} - Gục đầu xuống bàn")
                        st["logged_guc"] = True
            else:
                st["guc_start"] = None
                st["logged_guc"] = False

            # B. Hành vi Sử dụng điện thoại
            is_using_phone, matched_phone = kiem_tra_dien_thoai_gan_nguoi(phone_boxes, person_box, kpts)
            vi_pham_dt = False
            phone_dur = 0.0
            if is_using_phone:
                if st["phone_start"] is None:
                    st["phone_start"] = current_time
                phone_dur = current_time - st["phone_start"]
                if phone_dur >= tg_dien_thoai:
                    vi_pham_dt = True
                    ds_canh_bao.append(f"🟠 HS #{track_id}: SỬ DỤNG ĐIỆN THOẠI ({phone_dur:.1f}s)")
                    if not st["logged_phone"]:
                        trackers_state["history_log"].append(f"[{current_time:.1f}s] HS #{track_id} - Sử dụng điện thoại")
                        st["logged_phone"] = True
            else:
                st["phone_start"] = None
                st["logged_phone"] = False

            # C. Hành vi Giơ tay phát biểu (Chỉ xét khi không gục đầu)
            dang_gio_tay = False
            if not is_guc_dau:
                dang_gio_tay = phan_tich_gio_tay_phat_bieu(kpts)
                if dang_gio_tay:
                    ds_tich_cuc.append(f"🙋 HS #{track_id}: GIƠ TAY PHÁT BIỂU")
                    if not st["logged_giotay"]:
                        trackers_state["history_log"].append(f"[{current_time:.1f}s] HS #{track_id} - Giơ tay phát biểu")
                        st["logged_giotay"] = True
                else:
                    st["logged_giotay"] = False

            # Quyết định nhãn hiển thị & màu sắc khung
            if vi_pham_guc or vi_pham_dt:
                box_color = (0, 0, 255) # Đỏ: Vi phạm
                cac_nhan = []
                if vi_pham_guc:
                    cac_nhan.append(f"GUC DAU ({guc_dur:.1f}s)")
                if vi_pham_dt:
                    cac_nhan.append(f"DUNG DT ({phone_dur:.1f}s)")
                nhan = f"ID #{track_id}: " + " + ".join(cac_nhan)
            elif dang_gio_tay:
                box_color = (255, 160, 0) # Xanh dương sáng: Giơ tay phát biểu
                nhan = f"ID #{track_id}: GIO TAY PHAT BIEU"
            elif is_guc_dau or is_using_phone:
                box_color = (0, 215, 255) # Vàng: Đang theo dõi
                nhan = f"ID #{track_id}: Dang theo doi..."
            else:
                box_color = (0, 200, 0) # Xanh lá: Ngồi học nghiêm túc
                nhan = f"ID #{track_id}: Ngoi hoc nghiem tuc"

            overlay_items.append({
                "person_box": person_box,
                "kpts": kpts,
                "matched_phone": matched_phone if is_using_phone else None,
                "box_color": box_color,
                "nhan": nhan
            })

    return {
        "items": overlay_items,
        "ds_canh_bao": ds_canh_bao,
        "ds_tich_cuc": ds_tich_cuc
    }


def ve_ket_qua_len_frame(frame, cached_result):
    """Vẽ nhanh khung xương và nhãn hành vi lên frame mới nhất (tốn < 1ms CPU)"""
    annotated = frame.copy()
    if cached_result is None:
        return annotated

    h, w, _ = annotated.shape
    for item in cached_result["items"]:
        px1, py1, px2, py2 = item["person_box"]
        kpts = item["kpts"]
        box_color = item["box_color"]
        nhan = item["nhan"]

        # Vẽ khớp xương: Mũi, Mắt, Vai, Khuỷu tay, Cổ tay
        for kp_idx in [0, 1, 2, 5, 6, 7, 8, 9, 10]:
            if kp_idx < len(kpts) and kpts[kp_idx][0] > 0:
                cv2.circle(annotated, (int(kpts[kp_idx][0]), int(kpts[kp_idx][1])), 4, (0, 255, 128), -1)

        # Vẽ đường nối 2 vai và cánh tay
        if len(kpts) >= 11:
            if kpts[5][0] > 0 and kpts[6][0] > 0:
                cv2.line(annotated, (int(kpts[5][0]), int(kpts[5][1])), (int(kpts[6][0]), int(kpts[6][1])), (255, 220, 0), 2)
            if kpts[5][0] > 0 and kpts[7][0] > 0:
                cv2.line(annotated, (int(kpts[5][0]), int(kpts[5][1])), (int(kpts[7][0]), int(kpts[7][1])), (0, 255, 255), 2)
            if kpts[7][0] > 0 and kpts[9][0] > 0:
                cv2.line(annotated, (int(kpts[7][0]), int(kpts[7][1])), (int(kpts[9][0]), int(kpts[9][1])), (0, 255, 255), 2)
            if kpts[6][0] > 0 and kpts[8][0] > 0:
                cv2.line(annotated, (int(kpts[6][0]), int(kpts[6][1])), (int(kpts[8][0]), int(kpts[8][1])), (0, 255, 255), 2)
            if kpts[8][0] > 0 and kpts[10][0] > 0:
                cv2.line(annotated, (int(kpts[8][0]), int(kpts[8][1])), (int(kpts[10][0]), int(kpts[10][1])), (0, 255, 255), 2)

        # Vẽ khung điện thoại nếu có
        if item["matched_phone"] is not None:
            bx1, by1, bx2, by2, bconf = item["matched_phone"]
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (0, 140, 255), 2)
            cv2.putText(annotated, f"Dien thoai ({bconf:.2f})", (bx1, by1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 140, 255), 2)

        # Vẽ khung bao học sinh và nhãn trạng thái
        cv2.rectangle(annotated, (px1, py1), (px2, py2), box_color, 2)
        (tw, th), _ = cv2.getTextSize(nhan, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        cv2.rectangle(annotated, (px1, max(0, py1 - 26)), (px1 + tw + 10, py1), box_color, -1)
        cv2.putText(annotated, nhan, (px1 + 5, max(18, py1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

    # Vẽ thanh trạng thái tổng trên cùng
    ds_canh_bao = cached_result["ds_canh_bao"]
    ds_tich_cuc = cached_result["ds_tich_cuc"]

    if len(ds_canh_bao) > 0:
        cv2.rectangle(annotated, (0, 0), (w, 38), (0, 0, 180), -1)
        cv2.putText(annotated, "CANH BAO: " + " | ".join([c.split(":")[1] for c in ds_canh_bao]),
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    elif len(ds_tich_cuc) > 0:
        cv2.rectangle(annotated, (0, 0), (w, 38), (180, 100, 0), -1)
        cv2.putText(annotated, "PHAT BIEU: " + " | ".join([c.split(":")[1] for c in ds_tich_cuc]),
                    (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    else:
        cv2.rectangle(annotated, (0, 0), (w, 35), (20, 130, 20), -1)
        cv2.putText(annotated, "TRANG THAI LOP HOC: NGOI HOC NGHIEM TUC",
                    (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


# ==============================================================================
# 5. XỬ LÝ VIDEO TẢI LÊN HOẶC TỪ LINK
# ==============================================================================
def phan_tich_video_tong_hop(video_file, video_url, conf_phone, nguong_guc_dau, tg_guc_dau, tg_dien_thoai):
    duong_dan_video = None

    if video_url is not None and len(video_url.strip()) > 0:
        try:
            duong_dan_video = tai_video_tu_link(video_url)
            if duong_dan_video is None:
                return None, "❌ **Lỗi:** Không thể tải được video từ đường link này. Vui lòng kiểm tra lại link!"
        except Exception as e:
            return None, f"❌ **Lỗi khi tải video từ link:** `{str(e)}`"
    elif video_file is not None:
        duong_dan_video = video_file
    else:
        return None, "⚠️ **Vui lòng chọn 1 file video từ máy tính HOẶC dán đường link video vào ô bên trên!**"

    cap = cv2.VideoCapture(duong_dan_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or math.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = "ket_qua_video_ai.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    trackers_state = {"history_log": []}
    frame_idx = 0
    cached_result = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time = frame_idx / fps
        if frame_idx % 2 == 0 or cached_result is None:
            cached_result = suy_luan_ai_tren_frame(
                frame, current_time, trackers_state,
                conf_phone, nguong_guc_dau, tg_guc_dau, tg_dien_thoai
            )

        annotated_frame = ve_ket_qua_len_frame(frame, cached_result)
        out.write(annotated_frame)
        frame_idx += 1

    cap.release()
    out.release()

    logs = trackers_state["history_log"]
    nguon_video = f"Link URL ({video_url.strip()})" if (video_url and len(video_url.strip()) > 0) else "File tải lên từ máy"

    if len(logs) == 0:
        bao_cao = f"✅ **NGUỒN VIDEO:** {nguon_video}\n\n**KẾT QUẢ:** Học sinh ngồi học nghiêm túc, không có sự kiện bất thường."
    else:
        bao_cao = f"📊 **NGUỒN VIDEO:** {nguon_video}\n**GHI NHẬN {len(logs)} SỰ KIỆN HÀNH VI:**\n\n" + "\n".join([f"- {item}" for item in logs])

    return output_path, bao_cao


# ==============================================================================
# 6. CAMERA REAL-TIME SIÊU MƯỢT TRÊN HP ELITEBOOK 830 G5 (CACHED OVERLAY)
# ==============================================================================
dang_chay_camera = False

def bat_camera_hp830(conf_phone, nguong_guc_dau, tg_guc_dau, tg_dien_thoai):
    global dang_chay_camera
    dang_chay_camera = True

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    trackers_state = {"history_log": []}
    start_wall_time = time.time()
    frame_count = 0
    cached_result = None
    trang_thai_text = "### ✅ Đang giám sát thời gian thực..."

    while dang_chay_camera and cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time() - start_wall_time

        if frame_count % 3 == 0 or cached_result is None:
            cached_result = suy_luan_ai_tren_frame(
                frame, current_time, trackers_state,
                conf_phone, nguong_guc_dau, tg_guc_dau, tg_dien_thoai
            )

            ds_cb = cached_result["ds_canh_bao"]
            ds_tc = cached_result["ds_tich_cuc"]

            if len(ds_cb) > 0:
                trang_thai_text = "### 🚨 CẢNH BÁO VI PHẠM:\n" + "\n".join([f"- **{cb}**" for cb in ds_cb])
            elif len(ds_tc) > 0:
                trang_thai_text = "### 🙋 HOẠT ĐỘNG TÍCH CỰC:\n" + "\n".join([f"- **{tc}**" for tc in ds_tc])
            else:
                trang_thai_text = "### ✅ TRẠNG THÁI HIỆN TẠI:\n- Học sinh đang **Ngồi học nghiêm túc**."

            if len(trackers_state["history_log"]) > 0:
                trang_thai_text += "\n\n---\n**📋 Nhật ký hành vi gần nhất:**\n" + "\n".join([f"- {x}" for x in trackers_state["history_log"][-6:]])

        annotated_bgr = ve_ket_qua_len_frame(frame, cached_result)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        frame_count += 1
        yield annotated_rgb, trang_thai_text

    cap.release()


def tat_camera_hp830():
    global dang_chay_camera
    dang_chay_camera = False
    return None, "### ⏹️ Đã dừng Camera giám sát."


# ==============================================================================
# 7. GIAO DIỆN WEB DASHBOARD (CHUẨN 4 HÀNH VI)
# ==============================================================================
custom_css = """
.header-box {
    background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
    color: white;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    margin-bottom: 15px;
}
.header-box h1 { color: #ffffff !important; margin-bottom: 5px; font-size: 24px; font-weight: 800; }
.header-box p { color: #cbd5e1 !important; font-size: 14px; margin: 0; }
.status-panel {
    background-color: #f8fafc;
    border-left: 5px solid #1d4ed8;
    padding: 15px;
    border-radius: 8px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=custom_css, title="AI Giám Sát Học Sinh") as demo:
    gr.HTML("""
        <div class="header-box">
            <h1>🎓 HỆ THỐNG AI GIÁM SÁT HÀNH VI HỌC SINH TRONG LỚP HỌC</h1>
            <p>Nhận diện 4 hành vi: <b>Ngồi học nghiêm túc</b> | <b>Giơ tay phát biểu</b> | <b>Gục đầu xuống bàn</b> | <b>Sử dụng điện thoại</b></p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Tinh chỉnh Thông số AI")
            conf_phone_slider = gr.Slider(0.15, 0.80, value=0.30, step=0.05, label="Độ nhạy bắt Điện thoại (Confidence)")
            nguong_guc_slider = gr.Slider(
                0.15, 0.55, value=0.28, step=0.02,
                label="Ngưỡng Gục đầu xuống bàn (Ratio)",
                info="Mặc định 0.28: Chỉ báo khi đầu hạ sát xuống mặt bàn."
            )
            tg_guc_slider = gr.Slider(0.5, 8.0, value=2.0, step=0.5, label="Số giây xác nhận Gục đầu xuống bàn")
            tg_dt_slider = gr.Slider(0.5, 5.0, value=1.0, step=0.5, label="Số giây xác nhận Dùng điện thoại")
            gr.Markdown("""
            ---
            **💡 Chú thích khung màu AI:**
            * 🟢 **Xanh lá:** Ngồi học nghiêm túc
            * 🔵 **Xanh dương:** Giơ tay phát biểu
            * 🟡 **Vàng:** Đang theo dõi (mới cúi sát bàn / chạm ĐT)
            * 🔴 **Đỏ:** Vi phạm (Gục đầu xuống bàn / Dùng ĐT)
            """)

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("🔗 1. Phân tích Video (Tải File hoặc Gắn Link)"):
                    video_url_input = gr.Textbox(
                        label="🌐 Cách 1: Dán đường link Video (YouTube, TikTok, Facebook, hoặc link .mp4)",
                        placeholder="Ví dụ: https://www.youtube.com/watch?v=... hoặc https://vt.tiktok.com/...",
                        lines=1
                    )

                    with gr.Row():
                        video_file_input = gr.Video(label="📥 Cách 2: Hoặc tải file Video từ máy tính")
                        video_output = gr.Video(label="🎬 Video kết quả sau khi AI phân tích")

                    btn_analyze = gr.Button("🚀 BẮT ĐẦU TẢI & PHÂN TÍCH VIDEO", variant="primary", size="lg")
                    report_output = gr.Markdown("📋 *Nhật ký thời điểm các hành vi sẽ hiển thị tại đây...*", elem_classes=["status-panel"])

                    btn_analyze.click(
                        fn=phan_tich_video_tong_hop,
                        inputs=[video_file_input, video_url_input, conf_phone_slider, nguong_guc_slider, tg_guc_slider, tg_dt_slider],
                        outputs=[video_output, report_output]
                    )

                with gr.TabItem("📹 2. Camera Giám sát Thời gian thực (Real-time)"):
                    with gr.Row():
                        btn_start_cam = gr.Button("▶️ BẬT CAMERA GIÁM SÁT", variant="primary")
                        btn_stop_cam = gr.Button("⏹️ DỪNG CAMERA", variant="stop")

                    cam_output = gr.Image(label="🎯 Màn hình Giám sát AI Thời gian thực (Tối ưu 30 FPS)")
                    realtime_status = gr.Markdown("### ⏳ Nhấn nút 'BẬT CAMERA GIÁM SÁT' để bắt đầu...", elem_classes=["status-panel"])

                    btn_start_cam.click(
                        fn=bat_camera_hp830,
                        inputs=[conf_phone_slider, nguong_guc_slider, tg_guc_slider, tg_dt_slider],
                        outputs=[cam_output, realtime_status]
                    )
                    btn_stop_cam.click(
                        fn=tat_camera_hp830,
                        inputs=None,
                        outputs=[cam_output, realtime_status]
                    )

demo.launch(inbrowser=True)