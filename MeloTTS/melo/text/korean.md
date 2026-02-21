# korean.py 전체 흐름 정리

이 문서는 `melo/text/korean.py`의 입력/출력과 함수 간 데이터 흐름을 코드 기준으로 설명한다.

---

## 0. 파일의 역할

`korean.py`는 한국어 텍스트를 TTS 입력으로 바꾸는 모듈이다.

1. 텍스트 정규화 (`normalize`, `text_normalize`)
2. G2P 변환으로 phone 시퀀스 생성 (`korean_text_to_phonemes`, `g2p`)
3. BERT hidden state를 phone 길이에 맞춰 확장 (`get_bert_feature`)

최종적으로 TTS 인코더가 기대하는 형태인 다음 값을 만든다.

- `phones: List[str]`
- `tones: List[int]`
- `word2ph: List[int]`
- `bert_feature: torch.Tensor` with shape `[H, phone_len]`

---

## 1. 주요 전역 상태와 의존성

### 1-1. BERT 모델

- `BERT_MODEL_ID = "kykim/bert-kor-base"`
- `_models`, `_tokenizers` dict에 캐시한다.
- `_get_model_and_tokenizer()`가 최초 로딩 후 재사용한다.

### 1-2. G2P 모델

- `_g2p_kr` 전역에 `G2p()` 인스턴스를 지연 초기화(lazy init)한다.
- 최초 호출 시 로딩되고 이후 재사용한다.

### 1-3. 사전 기반 정규화

- `english_dictionary`, `etc_dictionary`로 치환한다.
- 영문 토큰과 약어 치환이 여기서 일어난다.

---

## 2. 함수별 입력/출력

## 2-1. `normalize_with_dictionary(text, dic)`

- 입력: 원문 문자열, 치환 dict
- 출력: dict key가 매칭된 부분 치환 문자열
- 특징: 정규식 OR 패턴으로 일괄 치환

## 2-2. `normalize_english(text)`

- 입력: 원문 문자열
- 출력: 영문 단어를 발음 사전 기반 한글 표기로 치환한 문자열
- 특징: 매칭 실패 단어는 원형 유지

## 2-3. `normalize(text)`

- 입력: 원문 문자열
- 출력: 한국어 정규화 문자열
- 처리:
  1) 앞뒤 공백 제거
  2) CJK 범위 일부 문자 제거
  3) `etc_dictionary` 치환
  4) 영문 치환
  5) 소문자화

## 2-4. `text_normalize(text)`

- 입력: 원문 문자열
- 출력: `normalize()` 결과
- 역할: 외부에서 부르는 정규화 진입점

## 2-5. `korean_text_to_phonemes(text, character="hangeul")`

- 입력: 텍스트 문자열
- 출력:
  - 기본(`hangeul`): 자모 시퀀스 문자열  
    예시) `"하늘"` -> `"하늘"`
  - `character="english"`: ascii 변환 문자열
- 처리:
  1) 정규화
  2) g2pkk 적용
  3) 기본 모드에서 `hangul_to_jamo` 적용

## 2-6. `distribute_phone(n_phone, n_word)`

- 입력:
  - `n_phone`: 단어에서 생성된 phone 개수
  - `n_word`: 해당 단어의 WordPiece 개수
- 출력: 길이 `n_word` 정수 리스트
- 의미: 각 WordPiece가 담당할 phone 개수 분배 (`word2ph` 구성용)

## 2-7. `_resolve_device(device)`

- 입력: device 문자열 또는 `None`
- 출력: 실제 사용 device 문자열
- 규칙:
  - macOS + mps 가능 + 입력 cpu면 mps
  - 입력 없음이면 cuda
  - cuda 요청했지만 불가하면 cpu로 강등

## 2-8. `_get_model_and_tokenizer(model_id, device)`

- 입력: model id, device
- 출력: `(model, tokenizer)`
- 특징:
  - 캐시 재사용
  - 모델을 요청 device로 이동

## 2-9. `g2p(norm_text, model_id)`

- 입력: 원문 텍스트
- 출력: `(phones, tones, word2ph)`
  - `phones: List[str]`
  - `tones: List[int]` (현재 전부 0)
  - `word2ph: List[int]`

핵심 보장:

- `len(word2ph) == len(tokenizer.tokenize(text_normalize(text))) + 2`
- `[CLS]`, `[SEP]` 대응으로 앞뒤 `1`을 추가한다.

## 2-10. `get_bert_feature(text, word2ph, device, model_id)`

- 입력:
  - 텍스트
  - `word2ph`
  - device
- 출력: `torch.Tensor` shape `[H, sum(word2ph)]`

처리:

1) `tokenizer(text, return_tensors="pt")`로 special token 포함 입력 생성  
2) `model(..., output_hidden_states=True)` 실행  
3) `hidden_states[-3]` 선택 -> `res: [T, H]`  
4) 각 `res[i]`를 `word2ph[i]`만큼 반복  
5) concat -> `[sum(word2ph), H]`  
6) transpose -> `[H, sum(word2ph)]`

---

## 3. `tokenize` vs `input_ids`

- `tokenizer.tokenize(text)` -> **문자열 토큰 리스트** (1차원)
- `tokenizer(text)["input_ids"]` -> **정수 ID 리스트** (1차원, special token 포함)

따라서 `for t in tokenized:`에서 `t`는 문자열이고 `startswith("#")`가 가능하다.

---

## 4. `ph_groups`의 상태: 2차원

`tokenized`는 1차원인데 `ph_groups`는 `List[List[str]]`다.

- `##`가 아닌 토큰: `ph_groups.append([t])` (새 그룹 시작)
- `##` 토큰: `ph_groups[-1].append(...)` (직전 그룹에 결합)

예:

- `tokenized = ["어", "##려", "##운", "데", "##이", "##터"]`
- `ph_groups = [["어", "려", "운"], ["데", "이", "터"]]`

---

## 5. `g2p()` 내부 상세 단계

1. `text_normalize()` 실행
2. tokenizer로 WordPiece 토큰화 (`tokenize`)
3. `##` 기준 그룹핑 (`ph_groups`)
4. 각 그룹을 문자열로 복원 (`piece_text = "".join(group)`)
5. 예외 처리:
   - `[UNK]` -> `phs += ["_"]`, `word2ph += [1]`
   - 문장부호 -> 그대로 `phs`에 추가, `word2ph += [1]`
6. 일반 토큰:
   - `korean_text_to_phonemes(piece_text)`로 자모열 생성
   - `distribute_phone(phone_len, word_len)`로 분배
   - `word2ph` 누적, `phs` 누적
7. 앞뒤 특수토큰 대응 추가:
   - `phones = ["_"] + phs + ["_"]`
   - `tones = [0] * len(phones)`
   - `word2ph = [1] + word2ph + [1]`
8. 길이 assert로 강제 정렬 검증

---

## 6. Shape 흐름 요약

`get_bert_feature()` 기준:

1. `inputs["input_ids"]`: 길이 `T`
2. `res = hidden_states[-3][0]`: shape `[T, H]`
3. 반복 확장 후: `[sum(word2ph), H]`
4. 반환 시 전치: `[H, sum(word2ph)]`

여기서 `T == len(word2ph)`가 반드시 성립해야 한다.

---

## 6-1. `phs += list(phonemes)`의 의미

- `phonemes`는 문자열이다.
- `list(phonemes)`는 문자열을 자모(음소) 단위 문자 리스트로 분해한다.
- 따라서 이 코드는 문자열 1개를 넣는 게 아니라, 자모 여러 개를 `phs`에 확장 추가한다.
- 예: `"하늘"` -> `["ᄒ","ᅡ","ᄂ","ᅳ","ᆯ"]`

---

## 6-2. 왜 `hidden_states[-3]`를 쓰는가

- BERT 출력 `hidden_states`는 임베딩 + 레이어별 은닉상태 튜플이다.
- `[-1]`은 마지막 레이어, `[-3]`은 뒤에서 3번째 레이어다.
- TTS 특징 추출에서는 마지막 레이어보다 `[-2]`, `[-3]`가 더 안정적인 경우가 많다.
- 현재 구현은 Melo 계열 기본 관례를 따라 `hidden_states[-3]`를 사용한다.
- 즉 필수 고정 규칙이라기보다 품질 경험치 기반 선택이다.

---

## 7. 왜 assert가 중요한가

검증식:

- `len(word2ph) == len(tokenized) + 2`
- `inputs["input_ids"].shape[-1] == len(word2ph)`

이게 깨지면:

- 토큰 임베딩 반복 인덱스가 어긋남
- phone-level feature 축 정렬 실패
- 품질 저하 또는 런타임 오류 가능

---

## 8. 실행 관점 입출력 예시

입력 텍스트:

- `"오늘은 날씨가 정말 좋네요."`

중간 산출물:

- `tokenized`: 문자열 WordPiece 리스트
- `ph_groups`: 단어 복원 그룹 리스트
- `word2ph`: 토큰별 반복 수 리스트

최종 산출물:

- `phones`: `["_", ...자모/문장부호..., "_"]`
- `tones`: `phones`와 동일 길이의 0 리스트
- `bert_feature`: `[H, len(phones)]`에 대응하는 phone-level feature
