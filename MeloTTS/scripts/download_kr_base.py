"""
Hugging Face에서 KR base checkpoint/config를 내려받아 pretrained/kr에 저장한다.
"""

from huggingface_hub import hf_hub_download
from melo.download_utils import LANG_TO_HF_REPO_ID
import os, shutil

BASE_DIR = "pretrained/kr"
os.makedirs(BASE_DIR, exist_ok=True)

LANG = "KR"
REPO_ID = LANG_TO_HF_REPO_ID[LANG]

# 1. checkpoint.pth 다운로드
ckpt_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="checkpoint.pth",
)

# 2. config.json 다운로드
cfg_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="config.json",
)

# 3. 로컬 폴더에 복사
shutil.copy(ckpt_path, f"{BASE_DIR}/checkpoint.pth")
shutil.copy(cfg_path, f"{BASE_DIR}/config.json")

print("KR base model saved to", BASE_DIR)
