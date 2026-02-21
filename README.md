# Kor_Voice_Lab

Kor_Voice_Lab은 한국어 음성 데이터 파이프라인(ASR -> 전처리 -> TTS 학습)과 MeloTTS 기반 추론/ONNX 변환 워크플로우를 정리한 프로젝트입니다.

## 0. 원본 리포지토리 및 라이선스

- 이 프로젝트의 TTS 베이스는 원본 [`MeloTTS` 리포지토리](https://github.com/myshell-ai/MeloTTS)를 기반으로 합니다.
- 원본 코드/가중치의 저작권 및 라이선스는 원본 MeloTTS 저장소 정책을 따릅니다.
- 로컬 커스텀 코드는 `Kor_Voice_Lab/MeloTTS` 하위에서 유지/수정/확장 중입니다.

## 0-1. 원본 MeloTTS 대비 변경점

- 한국어 파이프라인 대폭 수정:
- 한국어 텍스트 정규화/G2P/토큰 정렬/`word2ph` 경로를 한국어 BERT(예시: `kykim/bert-kor-base`) 기준으로 커스텀했습니다.
- 학습/추론/ONNX 변환 모두에서 KR 입력 feature(특히 BERT feature, `ja_bert` 경로)가 일관되게 맞도록 코드 경로를 개선했습니다.  
※ onnx는 해당 리포지토리의 확장이며, 현재 한국어만 지원합니다.

- 실행 안정성 개선:
- `python -m melo.infer`, `torchrun -m melo.train` 기준으로 실행 경로를 문서화하고 import 경로 이슈를 정리했습니다.

- ONNX 워크플로우 확장:
- `scripts/bert_onnx_converter.py`: BERT ONNX 변환 경로 정리(추론 품질 기준 설정 반영).
- `scripts/onnx_converter.py`: TTS ONNX 변환 흐름 정리.
- `scripts/infer_onnx.py`: KR 중심 torch-free ONNX 추론 경로 운영.
- `MeloTTS/tts_runtime`: 독립 ONNX 추론 런타임 구조를 추가해, 타 프로젝트 이식성을 확보했습니다.

## 0-2. 향후 계획 (GPT-SoVITS)

- 본 저장소는 향후 `GPT-SoVITS` 파이프라인을 추가해, MeloTTS와 병행 운영할 계획입니다.
- 목표는 동일 데이터셋/평가 기준으로 `MeloTTS`와 `GPT-SoVITS`를 비교 가능하게 구성하는 것입니다.
- README에는 GPT-SoVITS 추가 시 설치/학습/추론/모델 관리 절차를 별도 섹션으로 확장할 예정입니다.

## 1. 디렉토리 구조

- `asr_lab/`: Whisper 기반 전사 스크립트
- `preprocess/`: 리샘플링/메타데이터 생성 보조 스크립트
- `MeloTTS/`: TTS 학습/추론/ONNX 변환 코드

## 2. 환경 설치

### 2-1. 루트 의존성 설치

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2-2. MeloTTS 패키지 등록

`melo` import 에러를 막기 위해 editable 설치를 권장합니다.

```bash
cd /home/ahnhs2k/pytorch-demo/Kor_Voice_Lab/MeloTTS
uv pip install -e .
```

### 2-3. unidic 사전 다운로드

일본어 모듈 import 경로를 타는 환경에서는 MeCab/unidic 사전이 필요할 수 있습니다.

```bash
uv run python -m unidic download
```

## 3. ASR 전사 (Whisper)

`asr_lab/transcribe_whisper.py`로 오디오 폴더를 전사해서 MeloTTS용 메타데이터를 만듭니다.

```bash
cd /home/ahnhs2k/pytorch-demo/Kor_Voice_Lab
uv run asr_lab/transcribe_whisper.py \
  --audio_dir path/to/wavs \
  --out_metadata path/to/metadata.list \
  --speaker speaker \
  --language ko \
  --model large-v3 \
  --device auto \
  --compute_type auto \
  --vad \
  --beam_size 5 \
  --absolute_path
```

출력 포맷:

```text
/path/or/relative.wav|speaker_name|language_code|text
```

## 4. TTS 학습 데이터 준비

MeloTTS용 데이터 준비와 관련해서는 [MeloTTS 문서](MeloTTS/README.md) 참고하세요.
`pre

## 5. Pretrained 모델 설치

### (권장) 방법 A. KR base 수동 다운로드 스크립트

```bash
cd MeloTTS
uv run python scripts/download_kr_base.py
```

다운로드 위치:

- `MeloTTS/pretrained/kr/checkpoint.pth`
- `MeloTTS/pretrained/kr/config.json`

### 방법 B. 학습 시 자동 사용

`melo/train.py`는 `pretrain_G/D/DUR`를 주지 않으면 내부에서 기본 pretrained를 자동 로드 시도합니다.
다만, 한국어에 특화된 사전모델 다운로드로서 방법 A를 권장합니다.

## 6. TTS 학습

`MeloTTS` 디렉토리에서 실행합니다.

```bash
torchrun --nproc_per_node=1 --master_port=29501 \
  -m melo.train \
  -m outdir \
  -c path/to/config.json \
  --pretrain_G pretrained/kr/checkpoint.pth
```

체크포인트 저장 위치:

- `MeloTTS/logs/<model_name>/G_*.pth`
- `MeloTTS/logs/<model_name>/D_*.pth`
- `MeloTTS/logs/<model_name>/DUR_*.pth`

## 7. Torch 추론

```bash
uv run python -m melo.infer \
  -t "안녕하세요. 테스트 음성입니다." \
  -m logs/outdir/G_*.pth \
  -l KR \
  -o path/to/outdir
```

출력 WAV:

- `logs/outdir/<speaker_name>/output.wav`

## 8. ONNX 변환

### 8-1. BERT ONNX 변환

```bash
uv run python scripts/bert_onnx_converter.py \
  --model kykim/bert-kor-base \
  --out onnx_out/bert_kor.onnx \
  --device cuda
```

### 8-2. TTS ONNX 변환

```bash
uv run python scripts/onnx_converter.py \
  --config logs/model_dir/config.json \
  --ckpt logs/model_dir/G_*.pth \
  --out onnx_out/model.onnx \
  --device cuda \
  --text "오늘은 날씨가 정말 좋네요."
```

## 9. ONNX 추론

### 9-1. MeloTTS 내부 스크립트 사용

```bash
uv run python scripts/infer_onnx.py \
  --onnx onnx_out/model.onnx \
  --bert onnx_out/bert_kor.onnx \
  --config logs/model_dir/config.json \
  --text "드디어 온닉스 생성에 성공했습니다." \
  --speaker 0 \
  --lang KR \
  --device cpu \
  --out out_cpu.wav
```

### 9-2. 독립 런타임(`tts_runtime`) 사용

```bash
cd MeloTTS/tts_runtime
uv pip install -r requirements.txt
uv run infer_onnx.py \
  --onnx ../onnx_out/model.onnx \
  --bert ../onnx_out/model.onnx \
  --config ../logs/model_dir/config.json \
  --text "오늘은 날씨가 정말 좋네요." \
  --speaker 0 \
  --lang KR \
  --device cpu \
  --out out.wav
```

## 10. 자주 나는 오류와 원인

- `ModuleNotFoundError: No module named 'melo'`
- 원인: `MeloTTS` 패키지 설치 없이 파일 직접 실행 (`python melo/infer.py`)하거나 실행 위치가 잘못됨
- 해결: `uv pip install -e .` 후 `python -m melo.infer`, `torchrun -m melo.train` 사용

- `Failed initializing MeCab ... unidic/dicdir/mecabrc`
- 원인: unidic 사전 데이터 미설치
- 해결: `python -m unidic download`

- ONNX `tone_emb/Gather ... idx out of bounds`
- 원인: tone/language 매핑 불일치 또는 잘못된 symbols 설정
- 해결: 학습 config와 추론 파이프라인의 symbols/tone 맵 일치 확인

## 11. 실행 체크리스트

- `uv pip install -r requirements.txt` 완료
- `uv pip install -e MeloTTS` 완료
- 학습 전처리 산출물(`train.list`, `val.list`, `config.json`) 생성 완료
- pretrained 파일 위치 확인 (`MeloTTS/pretrained/kr/checkpoint.pth`)
- 추론은 항상 모듈 실행(`python -m ...`) 우선

## 인용

이 프로젝트가 유용했다면 아래 형식으로 인용해 주세요.

```bibtex
@misc{kor_voice_lab,
  title        = {Kor Voice Lab: Korean ASR-TTS Training, ONNX Conversion, and Runtime Pipeline},
  author       = {안호성 (GitHub: hobi2k)},
  year         = {2026},
  url          = {https://github.com/hobi2k/Kor_Voice_Lab},
  note         = {MeloTTS 기반 한국어 확장 및 ONNX 런타임 개선, Accessed: 2026-02-21}
}
```

## 참고 및 크레딧

- MeloTTS (원본): https://github.com/myshell-ai/MeloTTS
- Whisper / faster-whisper: https://github.com/SYSTRAN/faster-whisper
- Hugging Face `kykim/bert-kor-base`: https://huggingface.co/kykim/bert-kor-base
- ONNX Runtime: https://onnxruntime.ai/
