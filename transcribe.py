from faster_whisper import WhisperModel
import json 

model = WhisperModel("tiny", device="cpu", compute_type="int8")

segments, info = model.transcribe(
    "Calculus 1 Lecture 1.1：  An Introduction to Limits [54_XRjHhZzI].webm",
    word_timestamps=True
)

results = []
for segment in segments:
    results.append({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text
    })

with open("transcript.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved {len(results)} segments to transcript.json")