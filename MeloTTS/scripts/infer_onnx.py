"""
MeloTTS KR 모델을 위한 torch-free ONNX 추론 스크립트.

Examples:
  uv run python -m scripts.infer_onnx \
    --onnx onnx_out/saya/melo_saya.onnx \
    --bert onnx_out/bert_kor.onnx \
    --config onnx_out/saya/config.json \
    --text "드디어 온닉스 생성에 성공했습니다. 지연을 최소로 해서 거의 실시간 음성 합성이 가능해요." \
    --speaker 0 \
    --lang KR \
    --device cpu \
    --out test_cpu.wav
"""
import argparse
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import AutoTokenizer
from anyascii import anyascii
from jamo import hangul_to_jamo
from g2pkk import G2p

from melo.text import cleaned_text_to_sequence
from melo.text.symbols import punctuation
from melo.text.ko_dictionary import english_dictionary, etc_dictionary


# Minimal HParams (torch-free)
class HParams:
    """중첩 dict를 속성 접근 가능한 객체로 감싸는 간단한 컨테이너."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, dict):
                v = HParams(**v)
            self[k] = v

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)


def get_hparams_from_file(config_path: str) -> HParams:
    """config.json을 로드해 HParams 객체로 반환한다."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return HParams(**data)


# Korean text normalize + g2p (torch-free)
BERT_MODEL_ID = "kykim/bert-kor-base"
_tokenizers: Dict[str, AutoTokenizer] = {}


def _get_tokenizer(model_id: str) -> AutoTokenizer:
    """토크나이저를 캐시에서 재사용하고, 없으면 새로 로드한다."""
    if model_id not in _tokenizers:
        _tokenizers[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _tokenizers[model_id]


def _resolve_tokenizer_source(bert_onnx_path: Optional[str], fallback_model_id: str) -> str:
    """
    토크나이저 로드 경로를 결정한다.

    우선순위:
    1) --bert 로 전달된 ONNX 파일 폴더(bert_onnx_converter가 저장한 tokenizer 자산)
    2) fallback_model_id(HF repo id 또는 로컬 경로)
    """
    if bert_onnx_path:
        candidate = Path(bert_onnx_path).resolve().parent
        has_tokenizer_cfg = (candidate / "tokenizer_config.json").exists()
        has_vocab_file = (candidate / "tokenizer.json").exists() or (candidate / "vocab.txt").exists()
        if has_tokenizer_cfg and has_vocab_file:
            return str(candidate)
    return fallback_model_id


def normalize_with_dictionary(text: str, dic: Dict[str, str]) -> str:
    """치환 사전을 이용해 텍스트를 일괄 치환한다."""
    if any(key in text for key in dic.keys()):
        pattern = re.compile("|".join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    return text


def normalize_english(text: str) -> str:
    """영문 토큰을 한글 발음 사전에 따라 치환한다."""
    def fn(m: re.Match) -> str:
        word = m.group()
        if word in english_dictionary:
            return english_dictionary.get(word)
        return word

    return re.sub(r"([A-Za-z]+)", fn, text)


def text_normalize(text: str) -> str:
    """KR 추론용 텍스트 정규화(불필요 문자 제거/치환/소문자화)를 수행한다."""
    text = text.strip()
    text = re.sub(r"[\u2E80-\u2EF3\u2F00-\u2FD5\u3005\u3007\u3021-\u3029\u3038-\u303A\u303B\u3400-\u9FFF\uF900-\uFA6D]", "", text)
    text = normalize_with_dictionary(text, etc_dictionary)
    text = normalize_english(text)
    text = text.lower()
    return text


_g2p_kr = None


def korean_text_to_phonemes(text: str, character: str = "hangeul") -> str:
    """한국어 텍스트를 g2pkk+자모 분해로 phone 문자열로 변환한다."""
    global _g2p_kr
    if _g2p_kr is None:
        _g2p_kr = G2p()

    if character == "english":
        text = text_normalize(text)
        text = _g2p_kr(text)
        text = anyascii(text)
        return text

    text = text_normalize(text)
    text = _g2p_kr(text)
    text = list(hangul_to_jamo(text))
    return "".join(text)


def distribute_phone(n_phone: int, n_word: int) -> List[int]:
    """phone 개수를 WordPiece 조각 수에 최대한 균등 분배한다."""
    phones_per_word = [0] * n_word
    for _ in range(n_phone):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def g2p_kr(norm_text: str, model_id: str = BERT_MODEL_ID) -> Tuple[List[str], List[int], List[int]]:
    """정규화 텍스트를 KR 전용 phones/tones/word2ph로 변환한다."""
    tokenizer = _get_tokenizer(model_id=model_id)
    tokenized = tokenizer.tokenize(norm_text)

    phs: List[str] = []
    ph_groups: List[List[str]] = []

    for t in tokenized:
        if not t.startswith("#"):
            ph_groups.append([t])
        else:
            if not ph_groups:
                ph_groups.append([t.replace("#", "")])
            else:
                ph_groups[-1].append(t.replace("#", ""))

    word2ph: List[int] = []

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

        phonemes = korean_text_to_phonemes(piece_text)
        phone_len = len(phonemes)
        word_len = len(group)

        alloc = distribute_phone(phone_len, word_len)
        assert len(alloc) == word_len

        word2ph += alloc
        phs += list(phonemes)

    phones = ["_"] + phs + ["_"]
    tones = [0 for _ in phones]
    word2ph = [1] + word2ph + [1]

    assert len(word2ph) == len(tokenized) + 2, f"word2ph/token mismatch: {len(word2ph)}/{len(tokenized)+2}"

    return phones, tones, word2ph


def intersperse(lst: List[int], item: int) -> List[int]:
    """리스트 원소 사이사이에 blank 토큰을 삽입한다."""
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


# BERT ONNX (KR) -> phone-level feature
class BertOnnxRunner:
    """BERT ONNX를 실행해 phone-level feature를 생성하는 런타임 래퍼."""

    def __init__(self, onnx_path: str, provider: str, model_id: str = BERT_MODEL_ID):
        providers = ["CPUExecutionProvider"]
        if provider == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.tokenizer = _get_tokenizer(model_id=model_id)

    def run(self, norm_text: str, word2ph: List[int]) -> np.ndarray:
        """정규화 텍스트로 BERT ONNX를 실행하고 [H, T_phone] feature를 반환한다."""
        tokens = self.tokenizer(norm_text, return_tensors="np")
        input_ids = tokens["input_ids"].astype(np.int64)
        attention_mask = tokens["attention_mask"].astype(np.int64)

        if input_ids.shape[1] != len(word2ph):
            raise RuntimeError(f"input_ids len mismatch: {input_ids.shape[1]} vs {len(word2ph)}")

        outputs = self.session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            },
        )

        last_hidden = outputs[0]  # [B, T, H]
        res = last_hidden[0]  # [T, H]

        phone_level = []
        for i in range(len(word2ph)):
            phone_level.append(np.repeat(res[i:i + 1], word2ph[i], axis=0))
        phone_level = np.concatenate(phone_level, axis=0)  # [sum(word2ph), H]

        return phone_level.T  # [H, sum(word2ph)]


# ONNX TTS Inference (TEXT → AUDIO)
def infer_tts_onnx(
    onnx_path: str,
    bert_onnx_path: Optional[str],
    config_path: str,
    text: str,
    speaker_id: int,
    language: str = "KR",
    device: str = "cpu",
    noise_scale: float = 0.6,
    noise_scale_w: float = 0.8,
    length_scale: float = 1.0,
    out_path: str = "out.wav",
    bert_model_id: str = BERT_MODEL_ID,
):
    """
    텍스트를 ONNX 파이프라인(KR g2p -> BERT ONNX -> TTS ONNX)으로 합성한다.

    Returns:
        audio 1차원 numpy 배열
    """
    if language != "KR":
        raise NotImplementedError("Torch-free pipeline currently supports KR only.")

    hps = get_hparams_from_file(config_path)
    tokenizer_source = _resolve_tokenizer_source(bert_onnx_path, bert_model_id)
    symbols = hps.symbols
    symbol_to_id = {s: i for i, s in enumerate(symbols)}

    # 1. 텍스트 정규화 + g2p
    norm_text = text_normalize(text)
    phones, tones, word2ph = g2p_kr(norm_text, model_id=tokenizer_source)
    phones, tones, lang_ids = cleaned_text_to_sequence(phones, tones, language, symbol_to_id)

    # 2. add_blank 규칙 적용(학습/추론 일치)
    if getattr(hps.data, "add_blank", False):
        phones = intersperse(phones, 0)
        tones = intersperse(tones, 0)
        lang_ids = intersperse(lang_ids, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    # 3. BERT feature 구성(KR은 ja_bert 사용)
    if getattr(hps.data, "disable_bert", False):
        bert = np.zeros((1024, len(phones)), dtype=np.float32)
        ja_bert = np.zeros((768, len(phones)), dtype=np.float32)
    else:
        if not bert_onnx_path:
            raise ValueError("--bert is required unless hps.data.disable_bert is true")

        bert_runner = BertOnnxRunner(
            onnx_path=bert_onnx_path,
            provider=device,
            model_id=tokenizer_source,
        )
        bert_feat = bert_runner.run(norm_text, word2ph)
        ja_bert = bert_feat.astype(np.float32)
        bert = np.zeros((1024, ja_bert.shape[1]), dtype=np.float32)

    if bert.shape[1] != len(phones):
        raise RuntimeError(f"bert len mismatch: {bert.shape[1]} vs {len(phones)}")
    if ja_bert.shape[1] != len(phones):
        raise RuntimeError(f"ja_bert len mismatch: {ja_bert.shape[1]} vs {len(phones)}")

    # 4. 배치 차원 구성(B=1)
    x = np.array(phones, dtype=np.int64)[None, :]
    tone = np.array(tones, dtype=np.int64)[None, :]
    language_ids = np.array(lang_ids, dtype=np.int64)[None, :]
    x_lengths = np.array([x.shape[1]], dtype=np.int64)
    sid = np.array([speaker_id], dtype=np.int64)

    bert = bert[None, :, :].astype(np.float32)
    ja_bert = ja_bert[None, :, :].astype(np.float32)

    noise_scale = np.array(noise_scale, dtype=np.float32)
    noise_scale_w = np.array(noise_scale_w, dtype=np.float32)
    length_scale = np.array(length_scale, dtype=np.float32)

    # 5. ONNX Runtime 세션 생성(TTS)
    providers = ["CPUExecutionProvider"]
    if device == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    sess = ort.InferenceSession(onnx_path, providers=providers)

    # 6. 추론 실행(그래프에 없는 optional 입력은 자동 제외)
    input_names = {i.name for i in sess.get_inputs()}
    feed = {
        "x": x,
        "x_lengths": x_lengths,
        "sid": sid,
        "tone": tone,
        "language": language_ids,
        "bert": bert,
        "ja_bert": ja_bert,
        "noise_scale": noise_scale,
        "length_scale": length_scale,
        "noise_scale_w": noise_scale_w,
    }
    feed = {k: v for k, v in feed.items() if k in input_names}

    audio = sess.run(None, feed)[0]

    audio = audio[0, 0]

    # 7. wav 저장
    sr = hps.data.sampling_rate
    sf.write(out_path, audio, sr)
    print(f"[OK] saved → {out_path}")
    return audio

# CLI
def main():
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True, help="TTS ONNX model path")
    parser.add_argument("--bert", default=None, help="BERT ONNX model path (KR)")
    parser.add_argument(
        "--bert_model",
        default=BERT_MODEL_ID,
        help="fallback tokenizer source. if --bert folder has tokenizer files, that path is used first.",
    )
    parser.add_argument("--config", required=True, help="config.json path")
    parser.add_argument("--text", required=True)
    parser.add_argument("--speaker", type=int, default=0)
    parser.add_argument("--lang", default="KR")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", default="out.wav")

    parser.add_argument("--noise_scale", type=float, default=0.6)
    parser.add_argument("--noise_scale_w", type=float, default=0.8)
    parser.add_argument("--length_scale", type=float, default=1.0)

    args = parser.parse_args()

    infer_tts_onnx(
        onnx_path=args.onnx,
        bert_onnx_path=args.bert,
        config_path=args.config,
        text=args.text,
        speaker_id=args.speaker,
        language=args.lang,
        device=args.device,
        noise_scale=args.noise_scale,
        noise_scale_w=args.noise_scale_w,
        length_scale=args.length_scale,
        out_path=args.out,
        bert_model_id=args.bert_model,
    )


if __name__ == "__main__":
    main()
