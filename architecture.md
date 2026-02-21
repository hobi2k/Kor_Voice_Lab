# Kor_Voice_Lab 전체 아키텍처

이 문서는 `Kor_Voice_Lab`의 ASR -> 전처리 -> MeloTTS 학습/추론 -> ONNX 변환/추론까지 전체 구조를 코드 기준으로 정리한다.

## 1. 시스템 구성 개요

프로젝트는 크게 4개 계층으로 나뉜다.

1) 데이터 수집/정제  
- `preprocess/1.data_download.py`  
- `preprocess/2.resampling.py`  
- `preprocess/3.make_filelists.py`  
- `preprocess/4.policy_adapter.py`

2) ASR 전사  
- `asr_lab/transcribe_whisper.py`

3) MeloTTS 학습/파이토치 추론  
- `MeloTTS/melo/train.py`  
- `MeloTTS/melo/models.py`, `MeloTTS/melo/modules.py`, `MeloTTS/melo/attentions.py`  
- `MeloTTS/melo/infer.py`, `MeloTTS/melo/api.py`

4) ONNX 변환/ONNX 추론  
- `MeloTTS/scripts/bert_onnx_converter.py`  
- `MeloTTS/scripts/onnx_converter.py`  
- `MeloTTS/scripts/infer_onnx.py`  
- `MeloTTS/tts_runtime/infer_onnx.py` (독립 런타임 패키지)

---

## 2. 최상위 데이터 흐름

### 2-1. 학습 흐름

`원시 음성` -> `전사 metadata` -> `filelist` -> `config.json` -> `torchrun -m melo.train` -> `G/D/DUR 체크포인트`

### 2-2. 추론 흐름(PyTorch)

`텍스트` -> `text cleaner + g2p + BERT feature` -> `SynthesizerTrn` -> `wav`

### 2-3. ONNX 추론 흐름

`텍스트` -> `KR g2p + word2ph` -> `BERT ONNX(hidden_states[-3])` -> `TTS ONNX` -> `wav`

---

## 3. 디렉토리/모듈 역할

## 3-1. 루트 계층

- `asr_lab/`: faster-whisper 기반 전사
- `preprocess/`: 데이터셋 다운로드/리샘플링/메타 생성/필터링
- `MeloTTS/`: 원본 MeloTTS 기반 TTS 엔진 + 한국어 커스텀 + ONNX 확장
- `requirements.txt`: 루트 워크플로우 의존성(ASR/전처리/훈련 실행 환경)

## 3-2. MeloTTS 내부 핵심

- `melo/train.py`: 학습 엔트리포인트
- `melo/utils.py`: argparse/hparams/config 로딩, 로거/체크포인트 유틸
- `melo/data_utils.py`: filelist 기반 dataset/collate/sampler
- `melo/models.py`: 학습/추론용 메인 모델(SynthesizerTrn 등)
- `melo/modules.py`, `melo/attentions.py`, `melo/transforms.py`: 네트워크 블록/flow/attention
- `melo/losses.py`: adversarial + mel + KL + duration 관련 loss
- `melo/text/*`: 언어별 정규화, g2p, BERT feature 매핑
- `melo/mel_processing.py`: mel/spectrogram 변환

## 3-3. ONNX 계층

- `melo/models_onnx.py`: ONNX용 모델 구성(추론 그래프 전용)
- `melo/modules_onnx.py`: ONNX-safe 모듈 구현
- `melo/transforms_onnx.py`: ONNX-safe spline/flow 수학 연산
- `scripts/onnx_converter.py`: 체크포인트 -> TTS ONNX export
- `scripts/bert_onnx_converter.py`: HF BERT -> ONNX export
- `scripts/infer_onnx.py`: MeloTTS 내부 ONNX 추론 스크립트
- `tts_runtime/`: 외부 프로젝트 이식용 독립 ONNX 런타임

---

## 4. ASR 아키텍처 (`asr_lab/transcribe_whisper.py`)

## 4-1. 입력/출력 계약

- 입력:
  - `--audio_dir`: 오디오 폴더(재귀 탐색)
  - `--speaker`, `--language`
  - `--model`, `--device`, `--compute_type`, `--vad`, `--beam_size`
- 출력:
  - `metadata` 라인 포맷  
    `path|speaker|language|text`

## 4-2. 내부 처리 단계

1) 오디오 파일 수집 (`list_audio_files`)  
2) optional prompt 로드 (`read_text_file`)  
3) WhisperModel 생성 (모델 다운로드 위치: `asr_lab/pretrained`)  
4) 파일 단위 전사 (`transcribe_one`)  
   - VAD 사용 시 `min_silence_duration_ms=350`
5) 텍스트 정리 후 metadata 저장

---

## 5. 전처리 아키텍처 (`preprocess/*.py`)

## 5-1. `1.data_download.py`

- HuggingFace dataset 스트리밍 로드
- 조건 필터(language/speaker/transcription)
- audio bytes를 wav로 저장
- `metadata_raw.csv` 생성

## 5-2. `2.resampling.py`

- 입력 wav 일괄 로드
- target sample rate로 리샘플링
- mono + PCM_16으로 저장

## 5-3. `3.make_filelists.py`

- `metadata_raw.csv` 기반 `metadata.list` 생성
- 텍스트 최소 정규화(`|`, 개행, 다중 공백 정리)
- 한글 포함 여부 필터링
- 포맷: `wav|speaker|language|text`

## 5-4. `4.policy_adapter.py`

- 정책 기반 텍스트 필터/정제
  - 중괄호 메타, HTML, 괄호 주석, 특수 기호 제거
- wav 존재 검증
- 정제본 재작성 + `.bak` 백업 + `.dropped.txt` 로그 생성

---

## 6. MeloTTS 학습 아키텍처

## 6-1. 엔트리포인트와 실행 방식

- 엔트리포인트: `MeloTTS/melo/train.py`
- 권장 실행: `torchrun -m melo.train ...` (MeloTTS 루트에서)

## 6-2. 구성 로딩

- `utils.get_hparams()`가 다음을 결합
  - CLI 인자(`-m`, `-c`, `--pretrain_G` 등)
  - config json
  - 실행 디렉토리(`logs/<model_name>`)

## 6-3. 데이터 계층

- 학습/검증 리스트: `hps.data.training_files`, `hps.data.validation_files`
- 로더: `TextAudioSpeakerLoader`
- 배치: `TextAudioSpeakerCollate`
- 분산 샘플링: `DistributedBucketSampler`

## 6-4. 모델 계층

- Generator: `SynthesizerTrn`
- Discriminator: `MultiPeriodDiscriminator`
- Duration Discriminator: `DurationDiscriminator` (옵션)
- 보조 블록:
  - `attentions.py`의 encoder/attention
  - `modules.py`의 flow/coupling/conv blocks
  - `transforms.py`의 spline transform

## 6-5. 학습 루프/손실

- adversarial + feature matching + mel + KL + duration 관련 손실 조합
- AMP(`autocast`, `GradScaler`) 사용
- 체크포인트 주기 저장:
  - `logs/<model>/G_*.pth`, `D_*.pth`, `DUR_*.pth`

---

## 7. 텍스트/언어 처리 아키텍처

## 7-1. 언어별 모듈

- `melo/text/cleaner.py`가 언어별 모듈로 분기
- 한국어는 `melo/text/korean.py` 중심으로 커스텀

## 7-2. 한국어 경로 핵심

1) 텍스트 정규화  
2) BERT tokenizer WordPiece 토큰화  
3) `##` 그룹핑 후 음소화(g2pkk + jamo)  
4) `word2ph` 생성(토큰 축 -> phoneme 축 매핑)  
5) BERT `hidden_states[-3]`를 phone-level feature로 확장  
6) 모델 입력 shape `[H, phone_len]`로 정렬

자세한 설명: `MeloTTS/melo/text/korean.md`

---

## 8. ONNX 변환 아키텍처

## 8-1. BERT ONNX

- 스크립트: `MeloTTS/scripts/bert_onnx_converter.py`
- 역할: 한국어 BERT 출력(`hidden_states[-3]`)을 ONNX로 내보내기
- 목적: torch 없이도 text->BERT feature 생성

## 8-2. TTS ONNX

- 스크립트: `MeloTTS/scripts/onnx_converter.py`
- 역할:
  1) config + `G_*.pth` 로드
  2) `SynthesizerTrnONNX` 구성
  3) TTS 본체 ONNX export

의존 모듈:

- `melo/models_onnx.py` (상위 조립)
- `melo/modules_onnx.py` (ONNX-safe 블록)
- `melo/transforms_onnx.py` (ONNX-safe 변환)

---

## 9. ONNX 추론 아키텍처

## 9-1. 내부 스크립트

- `MeloTTS/scripts/infer_onnx.py`
- 흐름:
  1) config 로드
  2) KR g2p/word2ph 생성
  3) BERT ONNX 실행(선택)
  4) TTS ONNX 입력 feed 구성
  5) ORT 실행 후 wav 저장

## 9-2. 독립 런타임

- `MeloTTS/tts_runtime/`
- 목적:
  - `melo` 내부 폴더 의존 최소화
  - 다른 프로젝트로 이식 가능한 ONNX 추론 패키지 제공

---

## 10. 설정(Config) 아키텍처

## 10-1. 핵심 필드

- 데이터 경로: training/validation filelist
- 오디오 파라미터: sample rate, mel 관련 파라미터
- 모델 차원: hidden/channel/speaker/language/tone 크기
- 학습 파라미터: lr, batch size, epoch, fp16 등

## 10-2. 정렬 민감 항목

특히 아래 항목은 학습/추론/ONNX 간 불일치 시 오류나 품질 저하가 발생한다.

- `symbols` 순서/길이
- `num_tones`
- `num_languages`
- `data.spk2id`, `data.n_speakers`

---

## 11. 학습/추론 산출물(Artifacts)

## 11-1. 학습 산출물

- `logs/<model>/G_*.pth`
- `logs/<model>/D_*.pth`
- `logs/<model>/DUR_*.pth`
- `logs/<model>/config.json` (실행 시점 config 스냅샷)

## 11-2. ONNX 산출물

- BERT ONNX: `onnx_out/bert_kor.onnx`
- TTS ONNX: `onnx_out/<model>.onnx`
- 추론 wav: `out.wav` 또는 지정 출력 경로

---

## 12. 실행/배포 규칙 (권장)

1) MeloTTS 학습/torch 추론은 `MeloTTS` 루트에서 모듈 실행  
   - `torchrun -m melo.train ...`  
   - `python -m melo.infer ...`

2) ONNX 추론 배포는 `tts_runtime` 패키지 경로 사용

3) config-ckpt-onnx 세트는 동일 실험에서 나온 조합으로 고정

---

## 13. 현재 코드 상태 

- 사용 중:
  - `models_onnx.py`, `modules_onnx.py`, `transforms_onnx.py`


---

## 14. 향후 확장 지점

- GPT-SoVITS 파이프라인 추가 예정
- 토크나이저를 포함해서 onnx 런타임 완전 독립화

