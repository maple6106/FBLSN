
import os
import cv2
import numpy as np
from tqdm import tqdm
from mtcnn import MTCNN


# =========================
# 基本配置
# =========================
INPUT_ROOT = "ALLff++"
OUTPUT_ROOT = "FF++"

QUALITY = "c23"
FRAME_COUNT = 32
IMG_SIZE = 224

MANIP_METHODS = [
    "Deepfakes",
    "Face2Face",
    "FaceShifter",
    "FaceSwap",
    "NeuralTextures"
]

detector = MTCNN()


def extract_frames(video_path, n_frames=32):
    """从视频中等间隔抽取 n_frames 帧"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"⚠️ 无法打开视频：{video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    if total_frames < n_frames:
        frame_indices = np.linspace(0, total_frames - 1, total_frames).astype(int)
    else:
        frame_indices = np.linspace(0, total_frames - 1, n_frames).astype(int)

    frames = []

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()

        if ret and frame is not None:
            frames.append(frame)

    cap.release()
    return frames


def crop_face_mtcnn(frame):
    """使用 MTCNN 检测并裁剪最大人脸"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb)

    if len(detections) == 0:
        return None

    det = max(detections, key=lambda d: d["box"][2] * d["box"][3])
    x, y, w, h = det["box"]

    x = max(0, x)
    y = max(0, y)
    w = max(0, w)
    h = max(0, h)

    if w <= 0 or h <= 0:
        return None

    scale = 1.3
    cx = x + w // 2
    cy = y + h // 2
    size = int(max(w, h) * scale)

    x1 = max(0, cx - size // 2)
    y1 = max(0, cy - size // 2)
    x2 = min(frame.shape[1], cx + size // 2)
    y2 = min(frame.shape[0], cy + size // 2)

    crop = frame[y1:y2, x1:x2]

    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
    return crop


def process_video(video_path, save_dir):
    """处理单个视频：抽帧、人脸裁剪、保存图片"""
    os.makedirs(save_dir, exist_ok=True)

    frames = extract_frames(video_path, FRAME_COUNT)

    saved_count = 0

    for frame in frames:
        face = crop_face_mtcnn(frame)

        if face is None:
            continue

        saved_count += 1
        save_path = os.path.join(save_dir, f"img_{saved_count:04d}.jpg")
        cv2.imwrite(save_path, face)

    return saved_count


def process_original():
    """处理 original_sequences/youtube/c23/videos"""
    orig_dir = os.path.join(
        INPUT_ROOT,
        "original_sequences",
        "youtube",
        QUALITY,
        "videos"
    )

    if not os.path.exists(orig_dir):
        print(f"⚠️ 未找到原始视频目录：{orig_dir}")
        return

    videos = sorted([
        v for v in os.listdir(orig_dir)
        if v.lower().endswith(".mp4")
    ])

    print(f"\n开始处理 original，共 {len(videos)} 个视频")

    for vid in tqdm(videos, desc="original", ncols=120):
        video_path = os.path.join(orig_dir, vid)
        vid_name = os.path.splitext(vid)[0]

        save_dir = os.path.join(
            OUTPUT_ROOT,
            "original",
            vid_name
        )

        process_video(video_path, save_dir)


def process_manipulated():
    """处理 manipulated_sequences 下所有伪造方法"""
    manip_root = os.path.join(INPUT_ROOT, "manipulated_sequences")

    for method in MANIP_METHODS:
        videos_dir = os.path.join(
            manip_root,
            method,
            QUALITY,
            "videos"
        )

        if not os.path.exists(videos_dir):
            print(f"⚠️ 未找到 {method} 目录：{videos_dir}")
            continue

        videos = sorted([
            v for v in os.listdir(videos_dir)
            if v.lower().endswith(".mp4")
        ])

        print(f"\n开始处理 {method}，共 {len(videos)} 个视频")

        for vid in tqdm(videos, desc=method, ncols=120):
            video_path = os.path.join(videos_dir, vid)
            vid_name = os.path.splitext(vid)[0]

            save_dir = os.path.join(
                OUTPUT_ROOT,
                method,
                vid_name
            )

            process_video(video_path, save_dir)


def preprocess_ffpp_c23():
    print("=" * 60)
    print("开始预处理 FF++ c23 数据集")
    print(f"输入目录：{os.path.abspath(INPUT_ROOT)}")
    print(f"输出目录：{os.path.abspath(OUTPUT_ROOT)}")
    print("=" * 60)

    process_original()
    process_manipulated()

    print("\n🎉 FF++ c23 预处理完成！")
    print("输出目录：", os.path.abspath(OUTPUT_ROOT))


if __name__ == "__main__":
    preprocess_ffpp_c23()