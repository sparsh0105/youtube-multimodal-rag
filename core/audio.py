import whisper
from pathlib import Path
import math
import json

model = whisper.load_model("small.en")

# Audio extraction and transcription via whisper and then chunking 
def transcribe_and_chunk(video_id, window_size=10):
    out = Path(f"cache/transcripts/{video_id}.json")

    # If transcript already exists → reuse
    if out.exists():
        return json.loads(out.read_text())

    audio_path = f"cache/audio/{video_id}.wav"
    audio_file = Path(audio_path)
    
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    result = model.transcribe(audio_path)

    segments = result.get("segments", [])
    if not segments:
        return []

    last_end = segments[-1]["end"]
    total_windows = math.ceil(last_end / window_size)

    fixed_chunks = []

    for i in range(total_windows):
        window_start = i * window_size
        window_end = window_start + window_size

        texts = []
        for seg in segments:
            if seg["start"] < window_end and seg["end"] > window_start:
                texts.append(seg["text"].strip())

        combined_text = " ".join(texts).strip()
        if not combined_text:
            continue

        fixed_chunks.append({
            "start": window_start,
            "end": window_end,
            "audio_text": combined_text
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixed_chunks, ensure_ascii=False))

    return fixed_chunks
