import json

with open("transcript.json") as f:
    segments = json.load(f)

def chunk_by_time(segments, window_seconds=60):
    chunks = []
    current_chunk = {"start": segments[0]["start"], "end": None, "text": ""}

    for seg in segments:
        if seg["start"] - current_chunk["start"] > window_seconds:
            current_chunk["end"] = seg["start"]
            chunks.append(current_chunk)
            current_chunk = {"start": seg["start"], "end": None, "text": ""}
        current_chunk["text"] += seg["text"]

    current_chunk["end"] = segments[-1]["end"]
    chunks.append(current_chunk)
    return chunks

chunks = chunk_by_time(segments, window_seconds=60)

with open("chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)

print(f"Created {len(chunks)} chunks")