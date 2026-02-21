"""
uv run python melo/infer.py -t "데이터는 정말 심오해. 현기증이 날 것 같아." -m logs/yae_ko/G_35000.pth -l KR -o logs/yae_ko

uv run python -m melo.infer \
  -t "내가 누구냐고? 알 필요 없다." \
  -m logs/testing/G_30000.pth \
  -l KR \
  -o logs/testing
"""
import os
import click
from melo.api import TTS

    
    
@click.command()
@click.option('--ckpt_path', '-m', type=str, default=None, help="Path to the checkpoint file")
@click.option('--text', '-t', type=str, default=None, help="Text to speak")
@click.option('--language', '-l', type=str, default="KR", help="Language of the model")
@click.option('--output_dir', '-o', type=str, default="outputs", help="Path to the output")

def main(ckpt_path, text, language, output_dir):
    """
    체크포인트를 로드해 입력 텍스트를 화자별 wav 파일로 합성한다.

    Args:
        ckpt_path: 학습된 G 체크포인트 경로
        text: 합성할 입력 문장
        language: 언어 코드 (예: KR)
        output_dir: 출력 wav 저장 루트 디렉토리
    """
    if ckpt_path is None:
        raise ValueError("The model_path must be specified")

    config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
    model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)

    # 모델에 등록된 모든 화자(spk2id)에 대해 동일 텍스트를 합성한다.
    for spk_name, spk_id in model.hps.data.spk2id.items():
        save_path = f'{output_dir}/{spk_name}/output.wav'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.tts_to_file(text, spk_id, save_path)

if __name__ == "__main__":
    main()
