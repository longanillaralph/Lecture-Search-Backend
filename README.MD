# Lecture Search Tool

Mag-upload ng lecture recording, at makakakuha ka ng searchable transcript. Magtanong tungkol sa lecture sa normal na pananalita, at deretso kang makakapunta sa timestamp kung saan ito tinalakay — imbes na mag-scrub ng 75-minutong recording nang manual.

## Paano ito gumagana

Ang pipeline ay limang script, na pinapatakbo sa tamang pagkakasunod-sunod. Bawat isa ay nagbabasa ng file na ginawa ng script bago rito.

1. **I-download ang audio** — kumuha ng lecture recording mula sa YouTube gamit ang `yt-dlp` (o gumamit ng kahit anong audio/video file).
2. **`transcribe.py`** — ilo-load nito ang recording at ipapasa ito sa Whisper (`faster-whisper`) para makabuo ng timestamped transcript. Output: `transcript.json`, isang listahan ng maliliit na segments (kadalasan isang pangungusap bawat isa), bawat isa may start time, end time, at text.
3. **`chunk.py`** — pinagsasama-sama nito ang maliliit na segments na iyon papuntang mas malalaking piraso (mga 60 segundo bawat isa), para may sapat na context ang bawat piraso para maintindihan mag-isa. Output: `chunks.json` — mas kaunti ang laman kumpara sa `transcript.json`, pero mas mahaba ang bawat entry.
4. **`index.py`** — ginagawa nitong embedding (isang listahan ng numero na kumakatawan sa kahulugan) ang text ng bawat chunk, gamit ang sentence-transformers model, tapos itinatago ang embedding, ang orihinal na text, at ang timestamp nang magkakasama sa Chroma vector database. Output: isang `chroma_db/` folder sa disk (gumagamit ang Chroma ng SQLite sa ilalim para sa persistence, pero lahat ng read/write ay dumadaan sa Chroma's own API, hindi raw SQL).
5. **`search.py`** — ilo-load nito ang parehong embedding model, kukuha ng tanong na ini-type mo, gagawin itong embedding sa parehong paraan, tapos hihingi sa Chroma ng 3 chunks na pinakamalapit ang kahulugan sa tanong mo. Ipi-print ang timestamp range at text ng bawat resulta.

```
YouTube link
     |
     v
  yt-dlp  ---->  audio file (.webm/.mp3)
                       |
                       v
                transcribe.py  ---->  transcript.json
                       |
                       v
                  chunk.py      ---->  chunks.json
                       |
                       v
                  index.py      ---->  chroma_db/
                       |
                       v
                  search.py     ---->  top 3 relevant timestamps
```

## Bakit semantic search, hindi keyword search

Ang search ay hindi naghahanap ng exact word matches — nagma-match ito base sa *kahulugan*. Kapag tinanong mo ang "what is limits?", tama itong nag-return ng chunks tungkol sa left-side/right-side limits at limit existence, kahit na hindi literal na lumalabas ang salitang "limits" sa bawat match. Nangyayari ito dahil ang embeddings ay naglalagay ng text na magkatulad ang kahulugan nang magkalapit sa vector space, kahit magkaiba ang eksaktong mga salitang ginamit.

## Setup

```bash
uv venv --python 3.12
uv pip install faster-whisper sentence-transformers chromadb streamlit yt-dlp
```

## Paggamit

```bash
# 1. I-download ang isang lecture (o gumamit ng sarili mong audio/video file)
yt-dlp -x --audio-format mp3 "YOUTUBE_URL"

# 2. I-transcribe (i-edit muna ang filename sa loob ng transcribe.py)
python transcribe.py

# 3. I-chunk ang transcript
python chunk.py

# 4. Gawin ang searchable index
python index.py

# 5. Mag-search
python search.py
```

## Kasalukuyang status

- [x] Gumagana na ang transcription pipeline (na-test sa 75-minutong calculus lecture, 1618 raw segments)
- [x] Pag-chunk papuntang 60-segundong windows (82 chunks)
- [x] Embedding + Chroma indexing
- [x] Gumagana ang semantic search, tama at magkaibang resulta sa iba't ibang topic sa iisang lecture
- [ ] I-compare ang iba't ibang chunk window sizes gamit ang labeled na test questions, sukatin ang retrieval accuracy ng bawat size, piliin ang pinakamagaling na size base sa ebidensya
- [ ] Streamlit interface (upload + search box)
- [ ] I-deploy

## Stack

| Piraso | Tool |
|---|---|
| Transcription | faster-whisper (`tiny` model, CPU) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB |
| Interface (planned) | Streamlit |