import os
import re
import json
import torch
from pathlib import Path
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

device = "cuda" if torch.cuda.is_available() else "cpu"

# load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-large"
).to(device)

def get_timestamp(filename, interval_seconds):
    # OpenCV style → frame_0003_30s.jpg
    match = re.search(r"_(\d+)s\.jpg", filename)
    if match:
        return int(match.group(1))

    # FFmpeg style → frame_0013.jpg
    match = re.search(r"frame_(\d+)\.jpg", filename)
    if match:
        frame_number = int(match.group(1))
        return (frame_number - 1) * interval_seconds

    return None

def extract_visual_captions(video_id, interval_seconds=10):
    """
    Caption cached frames for a video and cache results.
    """
    captions_path = Path(f"cache/captions/{video_id}.json")
    frames_dir = Path(f"cache/frames/{video_id}")

    # If captions already exist → reuse
    if captions_path.exists():
        return json.loads(captions_path.read_text())

    if not frames_dir.exists():
        raise FileNotFoundError(
            f"No frames found for video_id={video_id}"
        )

    visual_captions = []

    for img_name in sorted(os.listdir(frames_dir)):
        if not img_name.lower().endswith(".jpg"):
            continue

        timestamp = get_timestamp(img_name, interval_seconds)
        if timestamp is None:
            continue

        image_path = frames_dir / img_name
        image = Image.open(image_path).convert("RGB")

        inputs = processor(image, return_tensors="pt").to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=50
            )

        caption = processor.decode(
            output[0],
            skip_special_tokens=True
        )

        visual_captions.append({
            "timestamp": timestamp,
            "caption": caption
        })

        print(f"{timestamp}s → {caption}")

    captions_path.parent.mkdir(parents=True, exist_ok=True)
    captions_path.write_text(
        json.dumps(visual_captions, ensure_ascii=False)
    )

    return visual_captions