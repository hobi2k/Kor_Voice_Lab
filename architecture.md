# Kor_Voice_Lab 아키텍처 명세

이 문서는 `Kor_Voice_Lab`를 구현 명세를 정리한 것이다.

---

## 0. 표기법

- `B`: batch size
- `T_text`: 텍스트 토큰 길이(phone 시퀀스 길이)
- `T_spec`: mel/spectrogram 프레임 길이
- `T_wav`: waveform 샘플 길이
- `H`: hidden channel
- `C`: latent/intermediate channel (`inter_channels`)

텐서 축 기준:
- 텍스트/잠재: `[B, C, T]`
- 파형: `[B, 1, T_wav]`
- BERT feature: `[B, H_bert, T_text]`

---

## 1. 시스템 분해

### 1-1. 파이프라인

1) ASR (`asr_lab/transcribe_whisper.py`)
2) 전처리 (`preprocess/*.py`)
3) TTS 학습 (`MeloTTS/melo/train.py`)
4) Torch 추론 (`MeloTTS/melo/infer.py`)
5) ONNX 변환
   - BERT ONNX (`MeloTTS/scripts/bert_onnx_converter.py`)
   - TTS ONNX (`MeloTTS/scripts/onnx_converter.py`)
6) ONNX 추론
   - 내부 스크립트 (`MeloTTS/scripts/infer_onnx.py`)
   - 독립 런타임 (`MeloTTS/tts_runtime/infer_onnx.py`)

### 1-2. 데이터 계약(공통)

학습/전처리 핵심 라인 포맷:

```text
wav_path|speaker_name|language_code|text
```

필수 조건:
- `wav_path`는 실제 파일 존재
- `speaker_name`은 config의 `spk2id`와 일치
- `language_code`는 `cleaner`/`text` 모듈이 인식 가능
- `text`는 비어 있지 않아야 함

---

## 2. 환경 재현 절차 (명령 고정)

### 2-1. 루트 환경

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 2-2. MeloTTS editable 설치

```bash
cd MeloTTS
uv pip install -e .
```

### 2-3. unidic(환경에 따라 필요)

```bash
uv run python -m unidic download
```

---

## 3. ASR 명세

파일: `asr_lab/transcribe_whisper.py`

입력 인자(핵심):
- `--audio_dir`
- `--out_metadata`
- `--speaker`
- `--language`
- `--model`, `--device`, `--compute_type`
- `--vad`, `--beam_size`, `--initial_prompt_file`
- `--absolute_path`

알고리즘:
1. `audio_dir` 재귀 탐색
2. 확장자 필터
3. Whisper 모델 로드 (`download_root=asr_lab/pretrained`)
4. 파일별 전사
5. 텍스트 최소 정리 후 metadata 저장

VAD 사용 시:
- `vad_filter=True`
- `min_silence_duration_ms=350`

산출물:
- metadata text 파일 (`path|spk|lang|text`)

---

## 4. 전처리 명세

### 4-1. `preprocess/1.data_download.py`

- HF dataset stream 로드
- 조건 필터(language/speaker/transcription)
- audio bytes를 `.wav`로 저장
- `metadata_raw.csv` 생성

### 4-2. `preprocess/2.resampling.py`

- 입력 wav를 target SR로 변환
- mono + PCM_16 저장

### 4-3. `preprocess/3.make_filelists.py`

- `metadata_raw.csv` -> `metadata.list`
- 텍스트 정규화(개행/`|`/다중 공백)
- 한국어 포함 여부 필터

### 4-4. `preprocess/4.policy_adapter.py`

- 정책 필터(메타 태그, HTML, 괄호, 특수기호 제거)
- wav 누락 라인 제거
- 백업(`.bak`) + drop 로그(`.dropped.txt`) 생성

---

## 5. 한국어 텍스트 경로 명세

핵심 파일:
- `MeloTTS/melo/text/korean.py`
- 상세 이론: `MeloTTS/melo/text/korean.md`

### 5-1. 핵심 함수 계약

#### `g2p(norm_text, model_id)`

입력:
- 문자열 텍스트

출력:
- `phones: List[str]`
- `tones: List[int]`
- `word2ph: List[int]`

동작:
1. 정규화
2. WordPiece 토큰화 (`tokenizer.tokenize`)
3. `##` 조각 그룹핑
4. 그룹 단위 g2pkk + jamo 변환
5. `distribute_phone`로 token->phone 반복수 분배
6. `[CLS]`, `[SEP]` 대응으로 `word2ph` 앞뒤 1 추가

불변조건:
- `len(word2ph) == len(tokenized) + 2`

#### `get_bert_feature(text, word2ph, device, model_id)`

입력:
- 텍스트
- `word2ph`

출력:
- `Tensor[H, sum(word2ph)]`

동작:
1. `tokenizer(text, return_tensors="pt")` (special token 포함)
2. `hidden_states[-3]` 선택
3. 각 토큰 feature를 `word2ph[i]`만큼 반복
4. concat 후 transpose

불변조건:
- `inputs['input_ids'].shape[-1] == len(word2ph)`

---

## 6. 학습 모델 구조 명세 (`MeloTTS/melo/models.py`)

## 6-1. `SynthesizerTrn` 구성

`SynthesizerTrn`는 아래 블록을 포함한다.

1. `enc_p: TextEncoder`
   - 입력: token/tone/language/bert/ja_bert
   - 출력: `x`, `m_p`, `logs_p`, `x_mask`

2. `enc_q: PosteriorEncoder`
   - 입력: 정답 spec
   - 출력: `z`, `m_q`, `logs_q`, `y_mask`

3. `flow: TransformerCouplingBlock` 또는 `ResidualCouplingBlock`
   - 역할: posterior 잠재를 prior 잠재 공간으로 가역 변환

4. duration 경로
   - `sdp: StochasticDurationPredictor`
   - `dp: DurationPredictor`

5. `dec: Generator`
   - latent -> waveform 복원

6. speaker conditioning
   - 다화자일 때 `emb_g`
   - 단일/참조 모드면 `ReferenceEncoder`

### 6-2. TextEncoder 입출력

입력:
- `x`: `[B, T_text]`
- `tone`: `[B, T_text]`
- `language`: `[B, T_text]`
- `bert`: `[B, 1024, T_text]`
- `ja_bert`: `[B, 768, T_text]`

출력:
- hidden: `[B, H, T_text]`
- `m_p`, `logs_p`: `[B, C, T_text]`
- `x_mask`: `[B, 1, T_text]`

### 6-3. PosteriorEncoder 입출력

입력:
- `y(spec)`: `[B, spec_channels, T_spec]`

출력:
- `z`, `m_q`, `logs_q`: `[B, C, T_spec]`
- `y_mask`: `[B, 1, T_spec]`

### 6-4. 학습 forward(핵심 순서)

`SynthesizerTrn.forward(...)`:

1. speaker embedding `g` 구성
2. `enc_p` 실행 -> `m_p`, `logs_p`
3. `enc_q` 실행 -> `z`, `m_q`, `logs_q`
4. `z_p = flow(z, y_mask, g)`
5. monotonic alignment(`maximum_path`) 계산
6. alignment로 `m_p/logs_p`를 frame 축으로 확장
7. random segment slice
8. `dec(z_slice, g)`로 파형 조각 생성

반환:
- `y_hat`, `l_length`, `attn`, `ids_slice`, masks,
- latent tuple `(z, z_p, m_p, logs_p, m_q, logs_q)`,
- duration tuple `(hidden_x, logw, logw_)`

### 6-5. 추론 forward(핵심 순서)

`SynthesizerTrn.infer(...)`:

1. `enc_p`
2. `logw = sdp(reverse=True)*sdp_ratio + dp*(1-sdp_ratio)`
3. `w = exp(logw) * x_mask * length_scale`
4. `w_ceil`로 경로 생성 (`generate_path`)
5. `m_p/logs_p`를 생성 길이로 확장
6. `z_p = m_p + randn * exp(logs_p) * noise_scale`
7. `z = flow(z_p, reverse=True)`
8. `o = dec(z)`

---

## 7. 학습 루프 명세 (`MeloTTS/melo/train.py`)

### 7-1. 배치 텐서 계약

데이터로더 출력:
- `x`, `x_lengths`
- `spec`, `spec_lengths`
- `y`, `y_lengths`
- `speakers`, `tone`, `language`
- `bert`, `ja_bert`

### 7-2. BERT 캐시 동작

`melo/data_utils.py:get_text` 기준:
- 캐시 경로: `<wav>.bert.pt`
- 존재 시 로드
- 없거나 shape mismatch면 생성 후 저장

즉 "학습 시 매 스텝 BERT 모델 호출"이 아니라,
- 최초 생성 후 캐시 재사용이 기본

### 7-3. 손실 계산 순서

1. Generator forward
2. Discriminator 업데이트(생성파형 detach)
3. Duration Discriminator 업데이트(옵션)
4. Generator 업데이트

사용 손실:
- `loss_disc`: 판별기 손실
- `loss_gen`: 생성기 adversarial 손실
- `loss_fm`: feature matching
- `loss_mel`: L1 mel
- `loss_dur`: duration 항
- `loss_kl`: KL(prior/posterior 정렬)
- duration discriminator generator 항(옵션)

---

## 8. ONNX 구조 명세

## 8-1. 왜 ONNX 전용 모듈이 필요한가

학습용 모듈(`modules.py`, `transforms.py`)은
- reverse/logdet/동적 분기/수치 경로가 섞여 있어 export 취약성이 높다.

그래서 ONNX 전용으로 분리:
- `melo/models_onnx.py`
- `melo/modules_onnx.py`
- `melo/transforms_onnx.py`

### 8-2. `SynthesizerTrnONNX` 입력 계약

입력 이름(onnx_converter 기준):
- `x`
- `x_lengths`
- `sid`
- `tone`
- `language`
- `bert`
- `ja_bert`
- `noise_scale`
- `length_scale`
- `noise_scale_w`

shape:
- `x`: `[B, T]`
- `x_lengths`: `[B]`
- `sid`: `[B]`
- `tone`, `language`: `[B, T]`
- `bert`: `[B, 1024, T]`
- `ja_bert`: `[B, 768, T]`
- scale 계열: scalar tensor

제약:
- 현재 구현은 B=1 경로 고정

### 8-3. ONNX forward 요약

1. `enc_p`
2. `dp`로 `logw`
3. `w_ceil`
4. `_expand_by_duration(m_p/logs_p)`
5. 샘플링 `z_p`
6. `flow_inv`
7. `dec`

---

## 9. ONNX 변환 명세

### 9-1. BERT ONNX 변환

파일: `MeloTTS/scripts/bert_onnx_converter.py`

역할:
- HF BERT를 ONNX로 export
- hidden state에서 `[-3]` 경로를 추론 입력으로 맞춤

주의:
- tokenizer 설정(예: max_length, truncation)은 더미 입력 shape에 영향

### 9-2. TTS ONNX 변환

파일: `MeloTTS/scripts/onnx_converter.py`

동작:
1. config 로드
2. `SynthesizerTrnONNX` 생성
3. 체크포인트 로드(키/shape 검증)
4. 더미 입력 생성
5. `torch.onnx.export` 실행

핵심 설정:
- `dynamo=False` (legacy exporter 경로)
- dynamic axes: text 길이 `T`, audio 길이 `T_audio`

---

## 10. 설정(config) 필수 항목

최소한 아래 항목은 반드시 맞아야 한다.

### 10-1. top-level
- `num_languages`
- `num_tones`
- `symbols`

### 10-2. data
- `training_files`, `validation_files`
- `sampling_rate`, `hop_length`, `filter_length`, `win_length`
- `n_mel_channels`, `mel_fmin`, `mel_fmax`
- `add_blank`
- `n_speakers`, `spk2id`

### 10-3. model
- `inter_channels`, `hidden_channels`, `filter_channels`
- `n_heads`, `n_layers`, `n_layers_trans_flow`
- `kernel_size`, `p_dropout`
- vocoder 파라미터(`resblock*`, `upsample*`)
- `gin_channels`
- duration/flow 사용 옵션

### 10-4. train
- `batch_size`, `learning_rate`, `epochs`
- `fp16_run`
- 손실 가중치(`c_mel`, `c_kl`)
- interval(`log_interval`, `eval_interval`)

---

## 11. 실패 포인트와 디버깅 규칙

### 11-1. `ModuleNotFoundError: melo`
- 원인: 실행 위치/방식 문제
- 해결: `MeloTTS` 루트에서 `python -m ...`, `torchrun -m ...`

### 11-2. ONNX `Gather idx out of bounds`
- 원인: tone/language/symbols 인덱스 불일치
- 점검:
  1) `num_tones`, `num_languages`
  2) `symbols` 길이/순서
  3) 추론 파이프라인의 tone/language 매핑

### 11-3. MeCab/unidic 오류
- 해결: `python -m unidic download`

### 11-4. 품질 저하(ONNX)
- 확인:
  1) BERT hidden layer 선택(`-1/-2/-3`)
  2) tokenizer 처리 일치(정규화/분절)
  3) export 옵션(opset/dynamic)
  4) config-ckpt-onnx 조합 동일성

---

## 12. 향후 확장

- GPT-SoVITS 파이프라인 추가
- 동일 데이터/평가 지표로 MeloTTS와 비교 실험 자동화
- tokenizer 자산까지 포함한 ONNX 런타임 완전 독립화

