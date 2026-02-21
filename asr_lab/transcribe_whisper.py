"""
transcribe_to_metadata.py

faster-whisper로 폴더 내 오디오 파일을 전사하고,
MeloTTS 학습용 metadata 파일을 생성한다.

출력 라인 포맷(중요: | 앞뒤 공백 없음)
path/to/audio.wav|speaker_name|language_code|text

형식 예시:
wavs/001.wav|yae_miko|ko|안녕하세요. 오늘은...

사용 예시:
uv run asr_lab/transcribe_whisper.py \
  --audio_dir /mnt/d//tts_data/yae_ko/wavs \
  --out_metadata /mnt/d/tts_data/yae_ko/asr_test/metadata.txt \
  --speaker yae_miko \
  --language ko \
  --model large-v3 \
  --device auto \
  --compute_type auto \
  --vad \
  --beam_size 5 \
  --absolute_path
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from faster_whisper import WhisperModel


# 지원 오디오 확장자
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac"}


def read_text_file(path: Optional[str]) -> str:
    """
    initial_prompt 파일을 읽고, 없으면 빈 문자열을 반환한다.

    프롬프트가 있을 시, 모델에 전달하여 고유명사나 인명 같은 단어의 오타를 줄인다.
    """
    if not path:
        return ""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"initial_prompt_file not found: {p}")

    return p.read_text(encoding="utf-8").strip()


def list_audio_files(audio_dir: str) -> List[Path]:
    """
    입력 폴더를 재귀 탐색해 오디오 파일 목록을 만든다.
    """
    root = Path(audio_dir)
    if not root.exists():
        raise FileNotFoundError(f"audio_dir not found: {root}")

    files: List[Path] = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS:
            files.append(p)
    return files


def clean_text(text: str) -> str:
    """
    전사 텍스트를 metadata 저장용으로 최소 정리한다.
    """
    t = (text or "").strip()
    t = t.replace("\n", " ").replace("\r", " ")
    while "  " in t:
        t = t.replace("  ", " ")
    return t


def transcribe_one(
    model: WhisperModel,
    audio_path: Path,
    language: Optional[str],
    initial_prompt: str,
    beam_size: int,
    use_vad: bool,
) -> str:
    """
    transcribe_one

    오디오 파일 1개를 faster-whisper로 전사하고,
    세그먼트 텍스트를 정리해 단일 문자열로 반환한다.

    Args:
        model: 로드된 WhisperModel 인스턴스
        audio_path: 전사할 오디오 파일 경로
        language: 전사 언어 코드(예: "ko"), 자동 감지 시 None
        initial_prompt: 고유명사/용어 힌트 문자열
        beam_size: beam search 크기 (음성 인식이 다음 단어를 고를 때, 1개만 바로 고르지 않고 상위 후보 여러 개를 동시에 유지하면서 최종 문장을 찾는 방법.)
        use_vad: 무음 구간 필터(VAD) 사용 여부

    Returns:
        정리된 최종 전사 문자열
    """

    # VAD는 무음이 많은 데이터에서 전사 안정성에 도움이 된다.
    # 350ms 이상의 무음은 구간 분리
    vad_params = {"min_silence_duration_ms": 350} if use_vad else None

    segments, _ = model.transcribe(
        str(audio_path),
        task="transcribe",
        language=language,
        beam_size=beam_size,
        initial_prompt=initial_prompt or None,
        vad_filter=use_vad,
        vad_parameters=vad_params,
    )

    # 세그먼트를 이어붙여 최종 문장을 만든다.
    parts: List[str] = []
    for seg in segments:
        s = clean_text(seg.text)
        if s:
            parts.append(s)

    return clean_text(" ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser()

    # 입출력
    parser.add_argument("--audio_dir", required=True, help="오디오 폴더 경로")
    parser.add_argument("--out_metadata", required=True, help="metadata 출력 파일 경로")

    # metadata 필드
    parser.add_argument("--speaker", required=True, help="speaker_name (예: yae_miko)")
    parser.add_argument("--language", default="ko", help="language_code (예: ko)")

    # Whisper 옵션
    parser.add_argument("--model", default="large-v3", help="tiny/base/small/medium/large-v3")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--compute_type", default="auto", help="auto/int8/float16/float32 등")

    # 전사 옵션
    parser.add_argument("--vad", action="store_true", help="VAD 사용(무음 필터)")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--initial_prompt_file", default=None, help="전사 힌트 텍스트 파일")

    # 경로 저장 방식
    parser.add_argument(
        "--absolute_path",
        action="store_true",
        help="metadata에 절대경로를 저장 (기본: audio_dir 기준 상대경로)",
    )

    args = parser.parse_args()

    audio_files = list_audio_files(args.audio_dir)
    if not audio_files:
        raise SystemExit(f"No audio files found in: {args.audio_dir}")

    out_path = Path(args.out_metadata)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    initial_prompt = read_text_file(args.initial_prompt_file)

    # ASR 모델 로컬 다운로드
    project_root = Path(__file__).resolve().parents[1]
    local_model_cache = project_root / "asr_lab" / "pretrained"
    local_model_cache.mkdir(parents=True, exist_ok=True)

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        download_root=str(local_model_cache),
    )

    audio_root = Path(args.audio_dir).resolve()

    with out_path.open("w", encoding="utf-8") as f:
        for idx, audio_path in enumerate(audio_files, start=1):
            print(f"[{idx}/{len(audio_files)}] transcribing: {audio_path.name}")

            text = transcribe_one(
                model=model,
                audio_path=audio_path,
                language=args.language,
                initial_prompt=initial_prompt,
                beam_size=args.beam_size,
                use_vad=args.vad,
            )

            # 빈 전사는 학습 품질에 악영향이 커서 제외한다.
            if not text:
                print("- skipped (empty transcript)")
                continue

            if args.absolute_path:
                path_str = str(audio_path.resolve())
            else:
                path_str = str(audio_path.resolve().relative_to(audio_root))

            line = f"{path_str}|{args.speaker}|{args.language}|{text}"
            f.write(line + "\n")

    print(f"\nDone. metadata saved: {out_path}")


if __name__ == "__main__":
    main()
