from pathlib import Path

import pandas as pd
from datasets import Audio, load_dataset
from tqdm import tqdm


# 저장 경로
OUT_DIR = Path("path/to/outdir")
AUDIO_DIR = OUT_DIR / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


# 데이터셋 로드
# - streaming=True: 전체를 한 번에 내려받지 않고 순차 처리
# - decode=False: 오디오를 bytes로 받아 torchcodec 의존을 줄임
dataset = load_dataset(
    "simon3000/genshin-voice",
    split="train",
    streaming=True,
)
dataset = dataset.cast_column("audio", Audio(decode=False))


# Korean + Yae Miko + 비어있지 않은 전사만 사용
dataset = dataset.filter(
    lambda v: (
        v["language"] == "Korean"
        and v["speaker"] == "Yae Miko"
        and v["transcription"] != ""
    )
)


# 오디오 저장 + 메타데이터 수집
rows = []
idx = 0

for item in tqdm(dataset):
    wav_path = AUDIO_DIR / f"{idx:05d}.wav"
    audio_bytes = item["audio"]["bytes"]

    # 일부 샘플은 bytes가 없을 수 있어 방어적으로 제외
    if audio_bytes is None:
        continue

    with open(wav_path, "wb") as f:
        f.write(audio_bytes)

    rows.append(
        {
            "wav": str(wav_path),
            "text": item["transcription"],
        }
    )
    idx += 1


# 메타데이터 저장
df = pd.DataFrame(rows)
df.to_csv(OUT_DIR / "metadata_raw.csv", index=False, encoding="utf-8")

print(f"[DONE] Saved {len(df)} Korean Paimon samples")
