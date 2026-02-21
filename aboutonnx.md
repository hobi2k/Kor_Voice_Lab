# ONNX 변환/구현 가이드

이 문서는 `Kor_Voice_Lab` 기준으로 ONNX를 왜 쓰는지, 무엇을 조심해야 하는지, 실제로 어떻게 변환/추론하는지 설명한다.

## 1. ONNX가 무엇인가

- ONNX는 모델을 프레임워크(PyTorch 등) 밖에서 실행할 수 있게 하는 **중간 표현 포맷**이다.
- 핵심 장점:
  - 배포 환경 단순화
  - CPU 추론 최적화(ONNX Runtime)
  - 다른 언어/플랫폼으로 이식 쉬움

중요: ONNX는 \"모델 가중치만 저장\"이 아니라 **연산 그래프 + 가중치**를 함께 저장한다.

---

## 2. 왜 이 프로젝트는 ONNX를 분리했나

`Kor_Voice_Lab/MeloTTS`는 ONNX를 두 파일로 나눈다.

1) BERT ONNX: 텍스트 -> BERT feature  
2) TTS ONNX: feature + text ids -> audio

이유:
- 학습용 MeloTTS 전체 경로를 단일 ONNX로 뽑으면 동적 정렬/분기/랜덤 연산 때문에 export가 불안정해지기 쉽다.
- 실무에서 안정적으로 운영하려면 “전처리/BERT”와 “TTS 본체”를 분리하는 방식이 훨씬 안전하다.

---

## 3. ONNX 구현 시 핵심 제약

## 3-1. 연산 제약

ONNX export가 깨지기 쉬운 패턴:
- 텐서 값에 의존하는 파이썬 분기
- boolean mask 기반 복잡 슬라이싱 후 함수 호출
- in-place 수정(특히 그래프 상수/경계값 처리)
- 런타임 assert/raise에 의존하는 forward 경로

그래서 이 프로젝트는 ONNX 전용 코드로 분리했다.
- `melo/models_onnx.py`
- `melo/modules_onnx.py`
- `melo/transforms_onnx.py`

## 3-2. shape 제약

이 프로젝트 ONNX 경로의 중요한 제약:
- 현재 `SynthesizerTrnONNX`는 **B=1** 경로를 전제로 함
- 입력 텐서 shape 계약을 반드시 지켜야 함
  - `x/tone/language`: `[B, T]`
  - `bert`: `[B, 1024, T]`
  - `ja_bert`: `[B, 768, T]`

## 3-3. config/가중치 제약

아래가 불일치하면 품질 저하/오류 가능:
- `symbols` 순서/길이
- `num_tones`
- `num_languages`
- `n_speakers`, `spk2id`

원칙: **onnx 변환에 쓰는 config는 반드시 해당 checkpoint와 같은 실험의 config를 사용**

---

## 4. 학습용 코드와 ONNX용 코드의 차이

- 학습용(`models.py`, `modules.py`, `transforms.py`)
  - 역전파/학습 보조 로직 포함
  - 다양한 분기/경로 포함

- ONNX용(`*_onnx.py`)
  - 추론 forward 경로 중심
  - export 안정성 중심
  - 불안정한 연산 패턴 제거

즉 ONNX용 코드는 \"학습 코드의 축소판\"이 아니라 **배포 목적에 맞춘 별도 구현**이다.

---

## 5. 실제 변환 절차

## 5-1. BERT ONNX 변환

```bash
cd MeloTTS
uv run python scripts/bert_onnx_converter.py \
  --model kykim/bert-kor-base \
  --out onnx_out/bert_kor.onnx \
  --device cuda
```

주의:
- 여기서 tokenizer의 `max_length`/`truncation`은 더미 입력 shape에 영향을 준다.
- 학습/추론 텍스트 처리 규칙과 최대한 일치시켜야 한다.

## 5-2. TTS ONNX 변환

```bash
uv run python scripts/onnx_converter.py \
  --config logs/<model>/config.json \
  --ckpt logs/<model>/G_XXXXX.pth \
  --out onnx_out/<model>.onnx \
  --device cuda \
  --text "더미 입력 문장"
```

핵심:
- checkpoint 키/shape를 검증해 안전 로드
- dynamic axes는 길이 축(`T`, `T_audio`) 중심으로 부여
- exporter 설정(`dynamo=False`)으로 안정성 확보

---

## 6. ONNX 추론 절차

```bash
uv run python scripts/infer_onnx.py \
  --onnx onnx_out/<model>.onnx \
  --bert onnx_out/bert_kor.onnx \
  --config logs/<model>/config.json \
  --text "안녕하세요." \
  --speaker 0 \
  --lang KR \
  --device cpu \
  --out out.wav
```

내부 흐름:
1) 한국어 g2p + `word2ph` 생성
2) BERT ONNX 실행(토큰 -> feature)
3) TTS ONNX 실행(feature -> audio)

---

## 7. 가장 자주 틀리는 포인트

1) **config/ckpt 혼용**
- 다른 실험 config를 쓰면 tone/lang/symbol mismatch가 발생하기 쉽다.

2) **tokenizer와 모델 구분 실패**
- BERT 모델을 ONNX로 바꿔도 tokenizer는 별도다.
- tokenizer 자산(vocab/tokenizer.json)을 함께 관리해야 한다.

3) **shape 계약 무시**
- `[B, T]`, `[B, C, T]` 축을 뒤섞으면 바로 오류/품질 저하.

4) **학습 코드 그대로 ONNX로 옮기려 함**
- ONNX용은 "단순하고 안전한 경로"로 재설계해야 한다.

---

## 8. 디버깅 체크리스트

- [ ] ONNX 입력 이름이 모델 그래프와 일치하는가
- [ ] `tone/language` id 범위가 embedding 크기 내에 있는가
- [ ] `symbols`, `num_tones`, `num_languages`가 config와 동일한가
- [ ] `len(word2ph)`와 BERT input length가 일치하는가
- [ ] 변환에 사용한 ckpt/config/onnx가 같은 실험 산출물인가

---

## 9. 한 줄 요약

ONNX 변환은 \"저장 포맷 변경\"이 아니라,  
**배포 가능한 추론 그래프로 재설계하는 작업**이다.  
이 프로젝트는 그 재설계를 `BERT ONNX + TTS ONNX` 분리 구조로 해결한다.

---

## 10. 용어 설명

### 10-1. ONNX / ORT / Export

- ONNX  
  - 의미: 모델 그래프+가중치를 저장하는 실행 포맷.
  - 필요 이유: PyTorch 없이 배포 가능.
  - 이 프로젝트: `bert_kor.onnx`, `melo_*.onnx` 두 개를 운영.

- ORT(ONNX Runtime)  
  - 의미: ONNX 실행 엔진.
  - 필요 이유: 실제 서비스 추론 담당(이식용).
  - 이 프로젝트: `scripts/infer_onnx.py`, `tts_runtime/infer_onnx.py`.

- export  
  - 의미: PyTorch 모델 -> ONNX 그래프 변환.
  - 자주 하는 실수: \"가중치 저장\" 정도로 생각하는 것.
  - 실제로는 연산 그래프 제약(분기/in-place/shape)까지 맞춰야 한다.

### 10-2. Trace / 더미 입력 / dynamic axes / opset

- trace(추적) + 더미 입력  
  - 의미: export할 때 모델을 한 번 실행해서 그래프를 기록.
  - 더미 입력은 그 실행에 쓰는 샘플 텐서.
  - 실수: 더미 입력이 너무 짧거나 특수해서 실제 입력과 괴리 발생.

- dynamic axes  
  - 의미: 길이 축(T)을 고정이 아니라 가변으로 남기는 설정.
  - 실수: dynamic axes를 줬다고 모든 동적 분기가 자동 해결된다고 착각.

- opset  
  - 의미: ONNX 연산자 규격 버전.
  - 실수: 낮은 opset을 고집하다 변환/실행 호환성 깨짐.

### 10-3. tokenizer / WordPiece / word2ph / phoneme

- tokenizer  
  - 의미: 텍스트를 토큰으로 분해하고 id로 변환.
  - 중요: BERT ONNX로 바꿔도 tokenizer는 별도 관리해야 함.

- WordPiece와 `##`  
  - 의미: 서브워드 분절. `##`는 앞 조각에 붙는 토큰.
  - 예: `["천", "##박", "##한"]`.

- `word2ph`  
  - 의미: 토큰 하나를 몇 개 음소(feature 반복)로 확장할지 나타내는 벡터.
  - 왜 필요: BERT 토큰 축과 음소 축 길이를 맞추기 위해.
  - 실수: 이 길이가 BERT input 길이와 안 맞으면 바로 정렬 붕괴.

- phoneme(음소)  
  - 의미: 발음 단위.
  - 한국어 경로에서는 자모열을 음소 축으로 사용.

### 10-4. prior / posterior / flow / reverse

- prior  
  - 텍스트 조건에서 예측한 잠재분포.

- posterior  
  - 정답 음성(spec)에서 인코딩한 잠재분포.

- flow  
  - 의미: 잠재변수를 가역적으로 변환하는 블록.
  - 역할: posterior 쪽 잠재를 prior 공간으로 정렬.

- reverse 분기(`reverse=True`)  
  - 의미: flow 역변환.
  - 학습: 정방향 사용 비중 큼.
  - 추론: 역방향으로 audio 생성 쪽 잠재 복원.
  - 코드 관점:
    - `reverse=False`: 정방향 \(x \rightarrow y\)
    - `reverse=True`: 역방향 \(y \rightarrow x\)
  - 왜 중요:
    - flow는 가역 변환을 전제로 하므로, 추론 시에는 역방향 경로가 실제 생성 경로가 된다.

### 10-5. rational-quadratic spline

- 한 줄 설명  
  - flow 안에서 쓰는 \"구간별 곡선 변환\".

- 왜 쓰는가  
  1) 선형보다 표현력이 높음  
  2) 단조 조건을 걸면 역함수 계산 가능  
  3) logdet 계산 가능(확률밀도 보정 가능)

- 비유  
  - 선형: 자 하나로 전체를 늘이거나 줄임  
  - spline: 구간마다 다른 곡선을 이어 더 정교하게 변형

- 코드 관점  
  - `widths/heights/derivatives`로 각 구간 모양 결정  
  - forward: x -> y  
  - inverse: y -> x

### 10-6. Jacobian / determinant / logdet

- Jacobian  
  - 벡터함수의 국소 미분 행렬.

- determinant  
  - 변환이 부피를 얼마나 늘리거나 줄였는지 나타내는 값.

- logdet(`log|det J|`)  
  - flow에서 변수변환 시 확률밀도 보정에 필요한 항.
  - 실수: logdet을 빼면 분포 학습이 틀어짐.

- logabsdet  
  - 코드에서 `log(|det|)`를 표현한 변수명.

### 10-7. in-place / mask slicing / clamp / gather / ellipsis

- in-place 연산  
  - 기존 텐서를 직접 수정.
  - ONNX에서 그래프 추적을 깨뜨릴 수 있어 회피 권장.

- boolean mask slicing  
  - 조건으로 텐서를 잘라 쓰는 방식.
  - 복잡한 형태는 export 비호환이 자주 난다.

- clamp  
  - 값을 범위로 제한.
  - 수치 안정성(예: 음수 판별식) 확보에 중요.

- gather  
  - 인덱스로 값 선택.
  - 인덱스 범위 초과 시 `idx out of bounds` 오류 발생.

- `...` (ellipsis)  
  - 앞 차원 전체를 의미하는 축약 인덱싱.
  - 예: `x[..., -1:]` = 마지막 축의 마지막 원소를 앞 차원 전체에 적용.

### 10-8. B=1 제약 / checkpoint

- B=1 제약  
  - 의미: ONNX 그래프를 단순화하기 위해 배치 1만 지원.
  - 실수: 배치 입력으로 바로 넣고 shape 오류 발생.

- checkpoint(`G_*.pth`)  
  - 학습된 가중치 파일.
  - ONNX 변환 시 이 파일을 로드해서 그래프 파라미터를 채운다.

---

## 11. 수학 상세: Flow / Spline / logdet

이 섹션은 \"왜 이런 식을 쓰는지\"를 구현 가능한 수준으로 정리한다.

### 11-1. 변수변환과 밀도 보정(핵심 공식)

가역 함수 \( y=f(x) \)에서 확률밀도는 아래를 만족한다.

\[
\log p_X(x) = \log p_Y(f(x)) + \log \left|\det \frac{\partial f(x)}{\partial x}\right|
\]

Flow에서는 여러 층 \(f_1, f_2, \dots, f_K\)를 합성하므로:

\[
z = f_K \circ \cdots \circ f_1(x),\quad
\log p_X(x)=\log p_Z(z)+\sum_{k=1}^{K}\log|\det J_{f_k}|
\]

요점:
- `logdet`은 옵션이 아니라 필수 보정항
- 수치 안정성 때문에 determinant 자체가 아니라 log-domain에서 누적

### 11-2. Coupling layer가 빠른 이유

입력을 채널 기준으로 \((x_a, x_b)\)로 나누고:

\[
y_a = x_a,\quad y_b = g(x_b; x_a)
\]

구조를 삼각 Jacobian 형태로 만들면 \(\det J\) 계산이 단순해진다.
- affine coupling이면 \(\logdet\)를 채널별 scale 합으로 빠르게 계산 가능
- 그래서 고차원에서도 학습/추론이 가능

### 11-3. Rational-Quadratic Spline(RQS) 기본 파라미터화

입력 구간 \([x_{\min}, x_{\max}]\)을 \(K\)개 bin으로 나눈다.
모델은 bin마다 아래 비정규화 파라미터를 예측:
- `unnormalized_widths`
- `unnormalized_heights`
- `unnormalized_derivatives`

정규화:
1) `softmax(widths)` -> 폭의 합이 1이 되게  
2) `softmax(heights)` -> 높이의 합이 1이 되게  
3) `softplus(derivatives)` -> 미분값 양수 강제  
4) 최소 제약:
   - `min_bin_width`
   - `min_bin_height`
   - `min_derivative`

이 제약이 필요한 이유:
- 너무 얇은 bin/기울기 0 근처를 막아 역함수/수치안정성 보장

### 11-4. RQS forward/inverse 개념

각 bin에서 곡선은 \"유리함수(rational)\" 형태로 정의된다.
실제 구현은 다음 흐름을 따른다.

forward:
1) 입력 \(x\)가 속한 bin index 찾기 (`searchsorted`)
2) bin 로컬 좌표 \(\theta\in[0,1]\) 계산
3) 분자/분모 다항식으로 \(y\) 계산
4) 동일 식으로 local derivative 계산 후 \(\logdet\) 산출

inverse:
1) \(y\)가 속한 bin 찾기
2) \(\theta\)에 대한 이차방정식 계수 \(a,b,c\) 구성
3) 판별식 \(b^2-4ac\)로 root 계산
4) \(\theta\to x\) 복원
5) 동일하게 \(\logdet\) 계산(역변환이라 부호 반전)

`transforms_onnx.py`의 inverse 분기에서
- `discriminant = clamp(discriminant, min=0)` 하는 이유:
  - 부동소수 오차로 아주 작은 음수가 생기는 케이스 방어

### 11-5. Tail 처리(선형 꼬리)

RQS는 주로 유한 구간에서 정의된다.
입력이 범위를 벗어날 때:
- 내부 구간: spline 적용
- 외부 구간: 항등함수(선형 tail)

즉 `unconstrained_rational_quadratic_spline`은
`torch.where(inside, spline, identity)` 형태로 동작한다.

이 방식의 장점:
- 도메인 밖에서도 폭주 없이 안정 동작
- ONNX에서 boolean slicing보다 그래프 안정적

---

## 12. 멜 스펙트로그램 상세

TTS 학습에서 mel은 \"정답 음성의 시간-주파수 표현\"으로 쓰인다.

### 12-1. 생성 절차(표준)

1) waveform \(x[n]\) 입력  
2) STFT 적용 (window, hop, FFT 길이)  
3) 파워 스펙트럼 \(|X_{t,f}|^2\)  
4) mel 필터뱅크 곱셈  
5) 로그/스케일 변환

결과 shape:
- 보통 `[n_mel_channels, T_spec]`

### 12-2. config와 직접 연결되는 파라미터

- `sampling_rate`
- `filter_length` (FFT 크기)
- `hop_length` (프레임 간격)
- `win_length` (윈도우 길이)
- `n_mel_channels`
- `mel_fmin`, `mel_fmax`

이 값들이 달라지면:
- mel 분포 자체가 바뀜
- 같은 모델이라도 추론 품질 급락 가능

즉 학습/추론/ONNX 모두 동일 파라미터를 유지해야 한다.

### 12-3. 왜 mel loss를 쓰는가

`train.py`에서 generator 학습 시 `L1(y_mel, y_hat_mel)`을 사용한다.

의미:
- 파형 샘플 단위보다 지각적으로 중요한 주파수 구조를 직접 맞춤
- GAN 손실만 쓸 때 생길 수 있는 음색 불안정을 줄임
- adversarial loss + feature matching + mel loss 조합이 안정적

### 12-4. segment 학습과 mel 정렬

학습은 긴 파형 전체가 아니라 segment를 잘라 학습한다.
- `ids_slice`로 waveform/spec 구간 동기화
- mel도 같은 구간으로 잘라서 비교

이 동기화가 틀어지면:
- mel loss가 잘못된 시간축을 비교하게 되어 학습이 망가진다

### 12-5. ONNX 추론에서 mel을 직접 안 쓰는 이유

ONNX 추론 경로는 보통 `text -> audio`만 수행한다.
mel은 학습 중 손실 계산/모니터링용이며,
추론에서는 decoder 출력 파형만 필요하다.

단, 품질 진단할 때는
- 추론 파형에서 mel을 다시 계산해
- 학습 분포와 비교 분석하는 것이 유효하다.
