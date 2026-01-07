# Load fix chunks from transcript cache

import json
from pathlib import Path

def load_fixed_chunks(video_id):
    path = Path(f"cache/transcripts/{video_id}.json")
    if not path.exists():
        raise FileNotFoundError("Fixed transcript chunks not found")
    return json.loads(path.read_text())

# Load visual captions from captions cache

def load_visual_captions(video_id):
    path = Path(f"cache/captions/{video_id}.json")
    if not path.exists():
        raise FileNotFoundError("Visual captions not found")
    return json.loads(path.read_text())

# Build multi-modal chunks from cached data 

def build_multimodal_chunks(video_id, fixed_chunks, visual_captions):
    """
    Merge fixed audio transcript chunks with visual captions
    and cache multimodal chunks.
    """

    out = Path(f"cache/chunks/{video_id}.json")
    if out.exists():
        return json.loads(out.read_text())

    # Build timestamp → caption map
    visual_map = {
        vc["timestamp"]: vc["caption"]
        for vc in visual_captions
    }

    multimodal_chunks = []

    for chunk in fixed_chunks:
        start = chunk["start"]
        end = chunk["end"]
        audio_text = chunk["audio_text"]

        visual_text = visual_map.get(start, "")

        combined_text = (
            f"Audio context:\n{audio_text}\n\n"
            f"Visual context:\n{visual_text}"
        )

        multimodal_chunks.append({
            "start": start,
            "end": end,
            "audio_text": audio_text,
            "visual_text": visual_text,
            "combined_text": combined_text
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(multimodal_chunks, ensure_ascii=False)
        )

    return multimodal_chunks

