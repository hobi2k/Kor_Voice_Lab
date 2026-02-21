import re
from pathlib import Path
from typing import List, Optional, Tuple


# 기존 설정
DATA_ROOT = Path("path/to/data")
WAV_DIR = DATA_ROOT / "wavs"
FILELIST_DIR = DATA_ROOT / "filelists"
META_PATH = FILELIST_DIR / "metadata.list"


def filter_and_clean_tts_text(text: str) -> Tuple[str, Optional[str]]:
    """
    TTS 학습용 텍스트를 필터링/정제한다.

    Returns:
        (status, cleaned_text)
        - status: KEEP | DROP | SKIP_EMPTY
    """
    # 1. 중괄호 메타가 있으면 제외
    if re.search(r"\{[^}]+\}", text):
        return "DROP", None

    # 2. HTML 태그 제거
    text = re.sub(r"<[^>]+>", "", text)

    # 3. 괄호 주석 제거
    text = re.sub(r"\([^)]*\)", "", text)

    # 4. 일본식/특수 따옴표/괄호 제거
    text = re.sub(r"[「」『』【】《》〈〉〔〕]", "", text)

    # 5. 특수 대시/물결 제거
    text = re.sub(r"[—–―~]", "", text)

    # 6. 기타 특수기호 제거
    text = text.replace("#", "")
    text = re.sub(r"[※★♪]", "", text)

    # 7. 공백 정리
    text = re.sub(r"\s+", " ", text).strip()

    # 8. 비어있으면 제외
    if not text:
        return "SKIP_EMPTY", None

    return "KEEP", text


def clean_filelist(filelist_path: Path, wav_dir: Path) -> None:
    """filelist를 정제하고 원본 백업(.bak) 및 제거 로그(.dropped.txt)를 남긴다."""
    wav_set = {p.name for p in wav_dir.glob("*.wav")}

    kept_lines: List[str] = []
    dropped_lines: List[str] = []

    with open(filelist_path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # 포맷 검증
            if "|" not in line:
                dropped_lines.append(f"{line}  # FORMAT_ERROR")
                continue

            wav, spk, lan, text = line.split("|")
            wav_name = Path(wav).name

            # wav 존재 검증
            if wav_name not in wav_set:
                dropped_lines.append(f"{line}  # WAV_NOT_FOUND")
                continue

            # 텍스트 정책 적용
            status, cleaned_text = filter_and_clean_tts_text(text)
            if status != "KEEP":
                dropped_lines.append(f"{line}  # TEXT_{status}")
                continue

            kept_lines.append(f"{wav}|{spk}|{lan}|{cleaned_text}")

    # 원본 백업 후 정제본 저장
    backup_path = filelist_path.with_suffix(filelist_path.suffix + ".bak")
    filelist_path.rename(backup_path)

    with open(filelist_path, "w", encoding="utf-8") as f:
        for line in kept_lines:
            f.write(line + "\n")

    # 제거 로그 저장
    drop_log = filelist_path.with_suffix(".dropped.txt")
    with open(drop_log, "w", encoding="utf-8") as f:
        for line in dropped_lines:
            f.write(line + "\n")

    print(f"[{filelist_path.name}] 정리 완료")
    print(f"  유지: {len(kept_lines)}")
    print(f"  제거: {len(dropped_lines)}")
    print(f"  백업: {backup_path.name}")
    print(f"  제거 로그: {drop_log.name}")


clean_filelist(META_PATH, WAV_DIR)
