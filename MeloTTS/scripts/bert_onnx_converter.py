"""
example:
uv run python scripts/bert_onnx_converter.py \
  --model pretrained/kr/bert-kor-base \
  --out onnx_out/bert_kor.onnx \
  --device cuda
"""

import argparse
from pathlib import Path
import torch
from transformers import BertModel, BertTokenizer


class BertONNXWrapper(torch.nn.Module):
    """
    MeloTTS 규칙과 맞춘 BERT ONNX 래퍼

    출력:
        hidden_states[-3] [B, T, H]
    """

    def __init__(self, model: BertModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-3]


def make_dummy_inputs(tokenizer, device):
    """ONNX 트레이싱용 더미 입력을 생성한다."""
    text = "오늘은 날씨가 정말 좋네요."
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        # onnx 모델이 일거에 더 많은 음성을 뽑을 수 있게 하려면 늘려야 한다.
        max_length=32,
    )

    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)

    return input_ids, attention_mask


def export_onnx(args):
    """HuggingFace BERT를 ONNX로 내보낸다."""
    device = torch.device(args.device)

    print("[INFO] loading tokenizer:", args.model)
    tokenizer = BertTokenizer.from_pretrained(args.model)

    print("[INFO] loading bert model:", args.model)
    bert = BertModel.from_pretrained(args.model)
    bert.eval()
    bert.to(device)

    onnx_model = BertONNXWrapper(bert).to(device)
    onnx_model.eval()

    dummy_inputs = make_dummy_inputs(tokenizer, device)

    print("[INFO] exporting BERT ONNX (hidden_states[-3])...")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        onnx_model,
        dummy_inputs,
        str(out_path),
        opset_version=18,
        input_names=["input_ids", "attention_mask"],
        output_names=["hidden_state_m3"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "hidden_state_m3": {0: "batch", 1: "seq_len"},
        },
        training=torch.onnx.TrainingMode.EVAL,
        do_constant_folding=True,
        dynamo=False,
    )

    # ONNX 모델 경로의 폴더에 tokenizer 자산도 같이 저장한다.
    tokenizer.save_pretrained(str(out_path.parent))
    print(f"[OK] BERT ONNX saved → {out_path}")
    print(f"[OK] tokenizer files saved → {out_path.parent}")


def main():
    """CLI 엔트리포인트."""
    parser = argparse.ArgumentParser("Export BERT to ONNX (SBV2-style)")
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--out", default="onnx_out/bert.onnx")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])

    args = parser.parse_args()
    export_onnx(args)


if __name__ == "__main__":
    main()
