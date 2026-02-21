"""
korean.py

1. g2p()가 만드는 word2ph 길이와 BERT inputs["input_ids"] 길이(CLS/SEP 포함) 완전 일치
2. 모델 구조상 get_bert_feature() 출력 구조를 "일본 코드와 완전히 동일"하게 유지
   - res: [T, H]
   - phone_level_feature: [sum(word2ph), H]
   - return: [H, sum(word2ph)]  (transpose)
3. Japanese bert 의존( .japanese_bert import ) 제거
4. BERT 모델은 한국어 모델로 고정 (kykim/bert-kor-base)
"""
from __future__ import annotations

import re
import sys
import unicodedata
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

from . import punctuation, symbols

from melo.text.ko_dictionary import english_dictionary, etc_dictionary
from anyascii import anyascii
from g2pkk import G2p
from jamo import hangul_to_jamo


def normalize_with_dictionary(text: str, dic: Dict[str, str]) -> str:
    """치환 사전에 있는 키를 우선순위 없이 일괄 치환한다."""
    if any(key in text for key in dic.keys()):
        pattern = re.compile("|".join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    return text


def normalize_english(text: str) -> str:
    """영문 토큰을 발음 사전 기반 한글 표기로 치환한다."""
    def fn(m: re.Match) -> str:
        word = m.group()
        if word in english_dictionary:
            return english_dictionary.get(word)
        return word

    text = re.sub(r"([A-Za-z]+)", fn, text)
    return text


# Text normalize (Korean)
def normalize(text: str) -> str:
    """한국어 텍스트를 사전 치환/영문 치환/소문자화 규칙으로 정규화한다."""
    text = text.strip()
    # CJK 제거(오리지날 코드 유지)
    text = re.sub(r"[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]", "", text)
    text = normalize_with_dictionary(text, etc_dictionary)
    text = normalize_english(text)
    text = text.lower()
    return text

def text_normalize(text: str) -> str:
    """
    외부에서 호출하는 한국어 정규화 진입점.
    
    clearner.py 참고
    """
    # 오리지널 코드 유지
    return normalize(text)

# Korean G2P (g2pkk + jamo)
_g2p_kr = None

def korean_text_to_phonemes(text: str, character: str = "hangeul") -> str:
    """
    텍스트 음소 변환 함수

    Args:
    - '하늘' (완성형)
    
    Returns:
    - '하늘' (자모 유니코드 시퀀스)
    """
    global _g2p_kr  # pylint: disable=global-statement
    if _g2p_kr is None:
        _g2p_kr = G2p()

    if character == "english":
        text = normalize(text)
        text = _g2p_kr(text)
        text = anyascii(text)
        return text

    text = normalize(text)
    text = _g2p_kr(text)
    text = list(hangul_to_jamo(text))
    return "".join(text)


def distribute_phone(n_phone: int, n_word: int) -> List[int]:
    """phone 개수를 WordPiece 조각 수만큼 최대한 균등하게 분배한다."""
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


# Korean BERT
# BERT 모델에 따라 수정이 필요해진다.
# 그 외 후보 BERT_MODEL_ID = "klue/bert-base"
BERT_MODEL_ID = "kykim/bert-kor-base"

_models: Dict[str, torch.nn.Module] = {}
_tokenizers: Dict[str, AutoTokenizer] = {}


def _resolve_device(device: Optional[str]) -> str:
    """입력 device 문자열을 현재 환경에서 유효한 장치로 정규화한다."""
    # 일본 코드 흐름을 최대한 유지
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        return "mps"
    if not device:
        return "cuda"
    # cuda 불가면 cpu로 강등
    if device.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return device


def _get_model_and_tokenizer(model_id: str, device: str):
    """모델/토크나이저를 캐시에서 재사용하고, 모델을 지정 장치로 이동한다."""
    if model_id not in _models:
        model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        _models[model_id] = model
        _tokenizers[model_id] = tokenizer
    else:
        model = _models[model_id]
        tokenizer = _tokenizers[model_id]

    # 항상 현재 device로 이동
    model = model.to(device)

    return model, tokenizer


# g2p (Korean): token alignment preserved
def g2p(norm_text: str, model_id: str = BERT_MODEL_ID) -> Tuple[List[str], List[int], List[int]]:
    """
    한국어 텍스트를 phone/tone/word2ph 시퀀스로 변환한다.

    핵심 보장:
    - tokenized = tokenizer.tokenize(norm_text)  (특수토큰 제외)
    - word2ph 길이 = len(tokenized) + 2  ([CLS], [SEP] 대응 1,1 추가)
    - get_bert_feature의 inputs["input_ids"] 길이도 동일
    """
    # 1. normalize (훈련/추론 동일하게)
    norm_text = text_normalize(norm_text)

    # 2. tokenizer 로드 (BERT와 동일 모델)
    device_dummy = "cpu"
    _, tokenizer = _get_model_and_tokenizer(model_id=model_id, device=device_dummy)

    # 3. 토큰화 (특수토큰 제외)
    tokenized = tokenizer.tokenize(norm_text)

    phs: List[str] = []
    ph_groups: List[List[str]] = []

    # WordPiece 그룹핑: "##" 붙은 조각들을 직전 토큰에 합친다.
    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            # 첫 토큰이 ##로 시작하는 이상 케이스
            if not ph_groups:
                ph_groups.append([t.replace("#", "")])
            else:
                ph_groups[-1].append(t.replace("#", ""))

    word2ph: List[int] = []

    # 4. 각 그룹을 하나의 "원형 토큰" 텍스트로 복원해서 g2pkk -> jamo
    for group in ph_groups:
        piece_text = "".join(group)

        if piece_text == "[UNK]":
            phs += ["_"]
            word2ph += [1]
            continue

        if piece_text in punctuation:
            phs += [piece_text]
            word2ph += [1]
            continue

        phonemes = korean_text_to_phonemes(piece_text)  # 자모 시퀀스 문자열
        phone_len = len(phonemes)
        word_len = len(group)

        # WordPiece 조각 수(word_len)에 phone_len을 분배
        aaa = distribute_phone(phone_len, word_len)
        # 원 코드의 안정성 체크(유지)
        assert len(aaa) == word_len

        word2ph += aaa
        # phonemes는 문자열이므로 list(phonemes)로 자모(음소) 단위로 펼쳐서 누적한다.
        # 예: "하늘" -> ["ᄒ","ᅡ","ᄂ","ᅳ","ᆯ"]
        phs += list(phonemes)

    # 5. [CLS], [SEP] 대응용 "_" 및 word2ph 1,1 추가
    phones = ["_"] + phs + ["_"]
    tones = [0 for _ in phones]
    word2ph = [1] + word2ph + [1]

    # 6. 일본 코드와 동일한 assert (토큰 수 정렬 강제)
    assert len(word2ph) == len(tokenized) + 2, f"word2ph/token mismatch: {len(word2ph)}/{len(tokenized)+2}"

    return phones, tones, word2ph


# get_bert_feature (Korean) — output structure identical to JP code
def get_bert_feature(
    text: str,
    word2ph: List[int],
    device: Optional[str] = None,
    model_id: str = BERT_MODEL_ID,
) -> torch.Tensor:
    """
    한국어 BERT hidden state를 phone 단위 feature로 확장해 반환한다.

    일본 코드와 출력 구조 동일:
    - inputs = tokenizer(text, return_tensors="pt")  (special tokens 포함)
    - res = hidden_states[-3] -> [T, H]
    - assert T == len(word2ph)
    - repeat로 phone-level 생성 -> [sum(word2ph), H]
    - return transpose -> [H, sum(word2ph)]
    """
    device = _resolve_device(device)
    text = text_normalize(text)

    model, tokenizer = _get_model_and_tokenizer(model_id=model_id, device=device)

    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")  # add_special_tokens 기본 True
        # tokenized = tokenizer.tokenize(text)  # 디버깅용(원 코드에 있으니 유지 가능)
        for k in inputs:
            inputs[k] = inputs[k].to(device)

        out = model(**inputs, output_hidden_states=True)
        hs = out["hidden_states"]

        # hidden_states[-3]를 사용한다.
        # Melo는 마지막 레이어(-1)보다 -3이 과도한 task-specific 성향이 덜해
        # 발화 특성(feature transfer)에서 더 안정적인 경우가 많아 사용한다.
        # (원 코드의 torch.cat(hs[-3:-2], -1)와 결과가 같다)
        res = hs[-3][0].cpu()  # [T, H]

    # 강제 정렬: token length == len(word2ph)
    assert inputs["input_ids"].shape[-1] == len(word2ph), f"{inputs['input_ids'].shape[-1]}/{len(word2ph)}"

    phone_level_feature: List[torch.Tensor] = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # [sum(word2ph), H]
    return phone_level_feature.T  # [H, sum(word2ph)]


# Standalone test
if __name__ == "__main__":
    test_text = "안녕하세요. 오늘 기분이 어때요?"
    phones, tones, word2ph = g2p(test_text, model_id=BERT_MODEL_ID)
    feat = get_bert_feature(test_text, word2ph, device="cpu", model_id=BERT_MODEL_ID)

    print("phones_len:", len(phones))
    print("sum(word2ph):", sum(word2ph))
    print("word2ph_len:", len(word2ph))
    print("bert_feature_shape [H, phone_len]:", tuple(feat.shape))
