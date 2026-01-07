import cv2
import subprocess
from pathlib import Path

# Extracting frames from video using OpenCV and FFmpeg for every 10 seconds
def extract_frames(video_id, interval_seconds=10):
    video_path = f"cache/video/{video_id}.mp4"
    video_file = Path(video_path)
    
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    out_dir = Path(f"cache/frames/{video_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if frames already exist
    existing_frames = list(out_dir.glob("*.jpg"))
    if existing_frames:
        print(f"✅ Using {len(existing_frames)} cached frames")
        return out_dir

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0

    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f}s")

    # Test OpenCV decoding
    ret, test_frame = cap.read()
    USE_OPENCV = ret and test_frame is not None
    cap.release()

    frame_interval = int(fps * interval_seconds)

    if USE_OPENCV:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                timestamp = int(frame_count / fps)
                filename = f"frame_{saved:04d}_{timestamp}s.jpg"
                cv2.imwrite(str(out_dir / filename), frame)
                saved += 1

            frame_count += 1

        cap.release()
        print(f"✅ Saved {saved} frames using OpenCV")

    else:
        print("🚀 OpenCV failed, using FFmpeg fallback")

        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps=1/{interval_seconds}",
            str(out_dir / "frame_%04d.jpg")
        ]

        subprocess.run(cmd, check=True)
        print("✅ Frames extracted using FFmpeg")

    return out_dir
