"""
KR base 체크포인트(config/checkpoint)와 한국어 BERT 자산을 로컬에 내려받는다.

기본 저장 경로:
- MeloTTS: pretrained/kr/{checkpoint.pth, config.json}
- BERT: pretrained/kr/bert-kor-base/*
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from melo.download_utils import LANG_TO_HF_REPO_ID


def download_melotts_kr(base_dir: Path) -> None:
    """MeloTTS KR base checkpoint/config를 내려받아 base_dir에 저장한다."""
    repo_id = LANG_TO_HF_REPO_ID["KR"]
    base_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = hf_hub_download(repo_id=repo_id, filename="checkpoint.pth")
    cfg_path = hf_hub_download(repo_id=repo_id, filename="config.json")

    shutil.copy(ckpt_path, base_dir / "checkpoint.pth")
    shutil.copy(cfg_path, base_dir / "config.json")
    print(f"[OK] MeloTTS KR base saved -> {base_dir}")


def download_bert_assets(bert_repo: str, bert_dir: Path) -> None:
    """
    한국어 BERT 모델/토크나이저를 로컬 디렉토리에 snapshot으로 저장한다.

    from_pretrained()가 로컬 경로를 바로 받을 수 있도록 전체 자산을 내려받는다.
    """
    bert_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=bert_repo,
        local_dir=str(bert_dir),
        local_dir_use_symlinks=False,
    )
    print(f"[OK] BERT assets saved -> {bert_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default="pretrained/kr",
        help="MeloTTS KR base 저장 경로 (default: pretrained/kr)",
    )
    parser.add_argument(
        "--bert_repo",
        default="kykim/bert-kor-base",
        help="다운로드할 BERT repo id (default: kykim/bert-kor-base)",
    )
    parser.add_argument(
        "--skip_bert",
        action="store_true",
        help="BERT 자산 다운로드를 생략한다.",
    )
    args = parser.parse_args()

    base_dir = Path(args.out_dir)
    download_melotts_kr(base_dir)

    if not args.skip_bert:
        download_bert_assets(args.bert_repo, base_dir / "bert-kor-base")


if __name__ == "__main__":
    main()
