"""
MeloTTS ONNX Exporter (Korean / kykim/bert-kor-base 기준 "최종본")

목표
- checkpoint(state_dict) "clean load" (shape mismatch 0)
- KR 파이프라인 규칙 준수:
    - get_text_for_tts_infer 로직 그대로:
      KR은 ja_bert에 BERT feature(768)를 넣고,
      bert(1024)는 zeros로 둔다.
- torch.export(dynamo) 피하고, legacy torch.onnx.export(트레이스)로 안정적으로 export

주의
- 본 스크립트는 "TextEncoderONNX가 bert_proj=1024, ja_bert_proj=768"인 구조를 전제로 한다.
- batch는 현실적으로 B=1을 표준으로 export한다.

  uv run python -m scripts.onnx_converter \
    --config logs/saya_ko/config.json \
    --ckpt logs/saya_ko/G_36000.pth \
    --out onnx_out/saya/melo_saya.onnx \
    --device cuda \
    --text "오늘은 날씨가 정말 좋네요."
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
import logging
from typing import Tuple, Dict, Any

import torch

# 스크립트 직접 실행(`python scripts/onnx_converter.py`) 시에도
# MeloTTS 루트를 import 경로에 추가해 `melo` 패키지를 찾게 한다.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from melo import commons
from melo.text import cleaned_text_to_sequence, get_bert
from melo.text.cleaner import clean_text
from melo.utils import HParams

# ONNX 전용 모델(TextEncoderONNX/Duration/Flow/Decoder 포함)
from melo.models_onnx import SynthesizerTrnONNX


logger = logging.getLogger("onnx_converter")


# 유틸: HParams 로드
def load_hparams_from_json(config_path: str) -> HParams:
    """config.json을 로드해 HParams 객체로 반환한다."""
    import json
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return HParams(**data)


# 유틸: checkpoint 로드
def load_checkpoint_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    """
    다양한 ckpt 포맷에서 state_dict를 추출한다.

    지원 포맷:
    - {"model": state_dict, ...}
    - state_dict 단독
    """
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt_obj, dict) and "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"]
    if isinstance(ckpt_obj, dict):
        # 혹시 키가 다를 수 있어도 일단 dict면 state_dict 취급
        # (단, optimizer 등 섞여있으면 깨질 수 있으니 model 키 우선)
        return ckpt_obj
    raise RuntimeError(f"Unsupported checkpoint format: {type(ckpt_obj)}")


# MeloTTS get_text_for_tts_infer 로직을 그대로 구현
def get_text_for_tts_infer_kr(
    text: str,
    language_str: str,
    hps: HParams,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      bert      : [1024, T]  (KR은 zeros)
      ja_bert   : [768,  T]  (KR은 kykim BERT feature)
      phone     : [T]
      tone      : [T]
      language  : [T]
    """
    # 1. cleaner
    norm_text, phone_list, tone_list, word2ph_list = clean_text(text, language_str)

    # 2. id 시퀀스화
    phone_list, tone_list, language_list = cleaned_text_to_sequence(phone_list, tone_list, language_str)

    # 3. blank intersperse 옵션 반영 (훈련/추론 일치 중요)
    if getattr(hps.data, "add_blank", False):
        phone_list = commons.intersperse(phone_list, 0)
        tone_list = commons.intersperse(tone_list, 0)
        language_list = commons.intersperse(language_list, 0)
        for i in range(len(word2ph_list)):
            word2ph_list[i] = word2ph_list[i] * 2
        word2ph_list[0] += 1

    # 4. BERT feature
    if getattr(hps.data, "disable_bert", False):
        bert_feature = torch.zeros(1024, len(phone_list), dtype=torch.float32)
        ja_bert_feature = torch.zeros(768, len(phone_list), dtype=torch.float32)
    else:
        bert_feature_raw = get_bert(norm_text, word2ph_list, language_str, str(device))  # [H, T]
        # get_text_for_tts_infer 원본 규칙:
        # - KR/JP/EN/... 은 ja_bert = bert, bert = zeros(1024)
        if language_str in ["JP", "EN", "ZH_MIX_EN", "KR", "SP", "ES", "FR", "DE", "RU"]:
            ja_bert_feature = bert_feature_raw.to(torch.float32)
            bert_feature = torch.zeros(1024, len(phone_list), dtype=torch.float32)
        elif language_str == "ZH":
            bert_feature = bert_feature_raw.to(torch.float32)
            ja_bert_feature = torch.zeros(768, len(phone_list), dtype=torch.float32)
        else:
            raise NotImplementedError(f"Unsupported language_str={language_str}")

    # 길이 검증
    if bert_feature.shape[-1] != len(phone_list):
        raise RuntimeError(f"bert_feature len mismatch: {bert_feature.shape[-1]} vs {len(phone_list)}")
    if ja_bert_feature.shape[-1] != len(phone_list):
        raise RuntimeError(f"ja_bert_feature len mismatch: {ja_bert_feature.shape[-1]} vs {len(phone_list)}")

    # tensor화
    phone_tensor = torch.LongTensor(phone_list)
    tone_tensor = torch.LongTensor(tone_list)
    language_tensor = torch.LongTensor(language_list)

    return bert_feature, ja_bert_feature, phone_tensor, tone_tensor, language_tensor


# 모델 준비 + clean load
def build_model_and_load(hps, ckpt_path: str, device: str):
    """ONNX용 모델을 생성하고 체크포인트를 shape-safe 방식으로 로드한다."""
    model = SynthesizerTrnONNX(hps).to(device)
    model.eval()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt

    model_sd = model.state_dict()

    # 1. flow_inv.* 는 체크포인트에 존재할 수 없는 파생 경로라 검증/로드에서 제외
    skip_prefixes = (
        "flow_inv.",  # inverse wrapper가 fwd를 submodule로 들고 있어서 생기는 중복 경로
    )

    # 2. 검증: 모델이 요구하는 키 중 skip_prefix 제외 항목이 체크포인트에 있는지 확인
    for k in model_sd.keys():
        if k.startswith(skip_prefixes):
            continue
        if k not in state:
            raise RuntimeError(f"[CKPT KEY MISSING] {k}")

    # 3. 로드: skip_prefix 제외 + shape 일치 키만 로드 (나머지는 초기값 유지)
    load_dict = {}
    for k, v in state.items():
        if k.startswith(skip_prefixes):
            continue
        if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape):
            load_dict[k] = v

    missing, unexpected = model.load_state_dict(load_dict, strict=False)

    # 디버그 출력
    if any(not m.startswith(skip_prefixes) for m in missing):
        real_missing = [m for m in missing if not m.startswith(skip_prefixes)]
        raise RuntimeError(f"[LOAD MISSING (unexpected)] {real_missing[:20]}")
    if unexpected:
        raise RuntimeError(f"[UNEXPECTED KEYS] {unexpected[:20]}")

    return model


# ONNX Export
def export_tts_onnx(
    config_path: str,
    ckpt_path: str,
    out_path: str,
    device_str: str,
    text: str,
    language_str: str = "KR",
    opset: int = 18,
):
    """TTS 본체를 ONNX로 export한다."""
    device = torch.device(device_str)
    hps = load_hparams_from_json(config_path)

    # (영어 변수명 + 한국어 의미) 스타일
    model_tts_onnx = build_model_and_load(hps, ckpt_path, device)

    # dummy input 생성 (MeloTTS 실제 infer 경로를 그대로 사용)
    bert_h_t, ja_bert_h_t, phone_t, tone_t, lang_t = \
        get_text_for_tts_infer_kr(text, language_str, hps, device)

    # 모델 forward 시그니처에 맞게 배치 차원을 추가하고 [B,C,T]로 정렬
    # phone/tone/lang: [T] -> [1,T]
    x_bt = phone_t.unsqueeze(0).to(device)
    tone_bt = tone_t.unsqueeze(0).to(device)
    lang_bt = lang_t.unsqueeze(0).to(device)

    # lengths: [1]
    x_len_b = torch.LongTensor([x_bt.size(1)]).to(device)

    # sid: [1] (일단 0번 화자)
    sid_b = torch.LongTensor([0]).to(device)

    # bert/ja_bert: [H,T] -> [1,H,T]
    bert_bht = bert_h_t.unsqueeze(0).to(device)
    ja_bert_bht = ja_bert_h_t.unsqueeze(0).to(device)

    # scalar params (B=1 고정)
    noise_scale_s = torch.tensor(0.667, dtype=torch.float32, device=device)
    length_scale_s = torch.tensor(1.0, dtype=torch.float32, device=device)
    noise_scale_w_s = torch.tensor(0.8, dtype=torch.float32, device=device)

    # ONNX export 준비
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    input_names = [
        "x", "x_lengths", "sid", "tone", "language", "bert", "ja_bert",
        "noise_scale", "length_scale", "noise_scale_w",
    ]
    output_names = ["audio"]

    # 동적 축(길이) 부여: T만 가변
    dynamic_axes = {
        "x": {1: "T"},
        "tone": {1: "T"},
        "language": {1: "T"},
        "bert": {2: "T"},
        "ja_bert": {2: "T"},
        "audio": {2: "T_audio"},  # 오디오 길이는 내부 duration에 의해 변함
    }

    logger.info(f"[INFO] Exporting TTS ONNX -> {out_path}")
    logger.info(f"[INFO] device={device} opset={opset} lang={language_str}")
    logger.info(f"[INFO] input T={x_bt.size(1)}")

    # 중요: dynamo=False로 legacy exporter를 강제해 export 안정성을 높인다.
    torch.onnx.export(
        model_tts_onnx,
        (
            x_bt,
            x_len_b,
            sid_b,
            tone_bt,
            lang_bt,
            bert_bht,
            ja_bert_bht,
            noise_scale_s,
            length_scale_s,
            noise_scale_w_s,
        ),
        out_path,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
        dynamo=False,
    )

    logger.info(f"[OK] TTS ONNX saved -> {out_path}")

    # ONNX와 동일 폴더에 config.json을 함께 복사해 배포 시 짝을 유지한다.
    out_dir = os.path.dirname(out_path) or "."
    config_copy_path = os.path.join(out_dir, "config.json")
    shutil.copy2(config_path, config_copy_path)
    logger.info(f"[OK] config copied -> {config_copy_path}")


def main():
    """CLI 엔트리포인트."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config.json path")
    parser.add_argument("--ckpt", type=str, required=True, help="G_XXXXX.pth path")
    parser.add_argument("--out", type=str, required=True, help="output onnx path")
    parser.add_argument("--device", type=str, default="cuda", help="cuda/cpu")
    parser.add_argument("--text", type=str, required=True, help="dummy text for tracing")
    parser.add_argument("--lang", type=str, default="KR", help="KR/JP/EN/ZH ... (default KR)")
    parser.add_argument("--opset", type=int, default=18, help="onnx opset (default 18)")
    args = parser.parse_args()

    export_tts_onnx(
        config_path=args.config,
        ckpt_path=args.ckpt,
        out_path=args.out,
        device_str=args.device,
        text=args.text,
        language_str=args.lang,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
