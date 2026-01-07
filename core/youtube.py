import subprocess
from pathlib import Path
from urllib.parse import urlparse, parse_qs



# Extract YouTube video ID from url

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)

    # youtu.be/<id>
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")

    # youtube.com/watch?v=<id>
    if "youtube.com" in parsed.netloc:
        query_params = parse_qs(parsed.query)
        if "v" in query_params and query_params["v"]:
            return query_params["v"][0]

    raise ValueError("Invalid YouTube URL")

# Downloading audio for audio extraction and transcription

def download_audio(video_id: str, url: str):
    out_dir = Path("cache/audio")
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / f"{video_id}.wav"
    if out.exists():
        return out

    subprocess.run(
        [
            "yt-dlp",
            "-x",
            "--audio-format", "wav",
            "-o", str(out_dir / "%(id)s.%(ext)s"),
            url
        ],
        check=True
    )

    return out

# Download video for frames extraction

def download_video(video_id: str, url: str):
    out_dir = Path("cache/video")
    out_dir.mkdir(parents=True, exist_ok=True)

    out = out_dir / f"{video_id}.mp4"
    if out.exists():
        return out

    subprocess.run(
        [
            "yt-dlp",
            "-f", "bestvideo+bestaudio/best",
            "--merge-output-format", "mp4",
            "-o", str(out_dir / "%(id)s.%(ext)s"),
            url
        ],
        check=True
    )

    return out
