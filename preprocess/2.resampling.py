from pathlib import Path

import librosa
import soundfile as sf
from tqdm import tqdm


# 입력/출력 경로 설정
SRC_DIR = Path("path/to/inputs")
DST_DIR = Path("path/to/outputs")
TARGET_SR = 44100

DST_DIR.mkdir(parents=True, exist_ok=True)


# 처리할 wav 목록 수집
wav_files = list(SRC_DIR.glob("*.wav"))
print(f"Found {len(wav_files)} wav files")


# 리샘플링 후 16-bit PCM으로 저장
for wav_path in tqdm(wav_files):
    try:
        # librosa.load에서 sr를 지정하면 자동 리샘플링된다.
        audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
        out_path = DST_DIR / wav_path.name
        sf.write(out_path, audio, TARGET_SR, subtype="PCM_16")
    except Exception as e:
        print(f"[SKIP] {wav_path.name} | {e}")

print("Resampling completed.")
