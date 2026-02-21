from pathlib import Path
import re

import pandas as pd


# 경로 설정
DATA_ROOT = Path("path/to/data")
WAV_DIR = DATA_ROOT / "wavs"
CSV_PATH = DATA_ROOT / "metadata_raw.csv"
FILELIST_DIR = DATA_ROOT / "filelists"
FILELIST_DIR.mkdir(parents=True, exist_ok=True)
META_PATH = FILELIST_DIR / "metadata.list"

SPEAKER = "saya_ko"
LANGUAGE = "KR"


def normalize_text(text: str) -> str:
    """파일리스트 저장 전 텍스트를 최소 정규화한다."""
    text = str(text)
    text = text.replace("|", " ")
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    return text


def has_korean(text: str) -> bool:
    """한글 포함 여부를 확인한다."""
    return bool(re.search(r"[가-힣]", text))


# CSV 로드
df = pd.read_csv(CSV_PATH)


# metadata.list 생성
lines = []
for _, row in df.iterrows():
    wav_name = Path(row["wav"]).name
    wav_path = WAV_DIR / wav_name

    # wav가 실제로 없으면 제외
    if not wav_path.exists():
        continue

    text = normalize_text(row["text"])
    if not text:
        continue

    # 한국어 샘플만 사용
    if not has_korean(text):
        continue

    lines.append(f"{wav_path}|{SPEAKER}|{LANGUAGE}|{text}")


META_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"meta: {len(lines)}")
