import os
import glob
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch
import torchaudio
import librosa
from melo.text import cleaned_text_to_sequence, get_bert
from melo.text.cleaner import clean_text
from melo import commons

MATPLOTLIB_FLAG = False

logger = logging.getLogger(__name__)



def get_text_for_tts_infer(text, language_str, hps, device, symbol_to_id=None):
    """
    추론용 원문 텍스트를 모델 입력 텐서로 변환한다.

    Returns:
        bert: [1024, T] 텐서(언어 설정에 따라 zero 텐서일 수 있음)
        ja_bert: [768, T] 텐서(언어 설정에 따라 zero 텐서일 수 있음)
        phone: [T] LongTensor
        tone: [T] LongTensor
        language: [T] LongTensor
    """
    norm_text, phone, tone, word2ph = clean_text(text, language_str)
    phone, tone, language = cleaned_text_to_sequence(phone, tone, language_str, symbol_to_id)

    if hps.data.add_blank:
        phone = commons.intersperse(phone, 0)
        tone = commons.intersperse(tone, 0)
        language = commons.intersperse(language, 0)
        for i in range(len(word2ph)):
            word2ph[i] = word2ph[i] * 2
        word2ph[0] += 1

    if getattr(hps.data, "disable_bert", False):
        bert = torch.zeros(1024, len(phone))
        ja_bert = torch.zeros(768, len(phone))
    else:
        bert = get_bert(norm_text, word2ph, language_str, device)
        del word2ph
        assert bert.shape[-1] == len(phone), phone

        if language_str == "ZH":
            bert = bert
            ja_bert = torch.zeros(768, len(phone))
        elif language_str in ["JP", "EN", "ZH_MIX_EN", 'KR', 'SP', 'ES', 'FR', 'DE', 'RU']:
            ja_bert = bert
            bert = torch.zeros(1024, len(phone))
        else:
            raise NotImplementedError()

    assert bert.shape[-1] == len(
        phone
    ), f"Bert seq len {bert.shape[-1]} != {len(phone)}"

    phone = torch.LongTensor(phone)
    tone = torch.LongTensor(tone)
    language = torch.LongTensor(language)
    return bert, ja_bert, phone, tone, language

def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    """체크포인트 파일에서 모델(선택적으로 옵티마이저) 상태를 로드한다."""
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    iteration = checkpoint_dict.get("iteration", 0)
    learning_rate = checkpoint_dict.get("learning_rate", 0.)
    if (
        optimizer is not None
        and not skip_optimizer
        and checkpoint_dict["optimizer"] is not None
    ):
        optimizer.load_state_dict(checkpoint_dict["optimizer"])
    elif optimizer is None and not skip_optimizer:
        # else:      Disable this line if Infer and resume checkpoint,then enable the line upper
        new_opt_dict = optimizer.state_dict()
        new_opt_dict_params = new_opt_dict["param_groups"][0]["params"]
        new_opt_dict["param_groups"] = checkpoint_dict["optimizer"]["param_groups"]
        new_opt_dict["param_groups"][0]["params"] = new_opt_dict_params
        optimizer.load_state_dict(new_opt_dict)

    saved_state_dict = checkpoint_dict["model"]
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            # assert "emb_g" not in k
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (
                saved_state_dict[k].shape,
                v.shape,
            )
        except Exception as e:
            print(e)
            # For upgrading from the old version
            if "ja_bert_proj" in k:
                v = torch.zeros_like(v)
                logger.warn(
                    f"Seems you are using the old version of the model, the {k} is automatically set to zero for backward compatibility"
                )
            else:
                logger.error(f"{k} is not in the checkpoint")

            new_state_dict[k] = v

    if hasattr(model, "module"):
        model.module.load_state_dict(new_state_dict, strict=False)
    else:
        model.load_state_dict(new_state_dict, strict=False)

    logger.info(
        "Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration)
    )

    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    """모델과 옵티마이저 상태를 체크포인트 파일로 저장한다."""
    logger.info(
        "Saving model and optimizer state at iteration {} to {}".format(
            iteration, checkpoint_path
        )
    )
    if hasattr(model, "module"):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(
        {
            "model": state_dict,
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "learning_rate": learning_rate,
        },
        checkpoint_path,
    )


def summarize(
    writer,
    global_step,
    scalars={},
    histograms={},
    images={},
    audios={},
    audio_sampling_rate=22050,
):
    """tensorboard에 scalar/histogram/image/audio 요약을 기록한다."""
    for k, v in scalars.items():
        writer.add_scalar(k, v, global_step)
    for k, v in histograms.items():
        writer.add_histogram(k, v, global_step)
    for k, v in images.items():
        writer.add_image(k, v, global_step, dataformats="HWC")
    for k, v in audios.items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    """숫자 접미사 기준으로 가장 최신 체크포인트 경로를 반환한다."""
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    return x


def plot_spectrogram_to_numpy(spectrogram):
    """스펙트로그램 배열을 RGB numpy 이미지로 변환한다."""
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation="none")
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def plot_alignment_to_numpy(alignment, info=None):
    """어텐션 정렬 행렬을 RGB numpy 이미지로 변환한다."""
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib

        matplotlib.use("Agg")
        MATPLOTLIB_FLAG = True
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
    import matplotlib.pylab as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(
        alignment.transpose(), aspect="auto", origin="lower", interpolation="none"
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    plt.tight_layout()

    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return data


def load_wav_to_torch(full_path):
    """scipy로 wav를 로드해 float 텐서와 샘플레이트를 반환한다."""
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_wav_to_torch_new(full_path):
    """torchaudio로 wav를 로드하고 mono로 다운믹스한다."""
    audio_norm, sampling_rate = torchaudio.load(full_path, frame_offset=0, num_frames=-1, normalize=True, channels_first=True)
    audio_norm = audio_norm.mean(dim=0)
    return audio_norm, sampling_rate

def load_wav_to_torch_librosa(full_path, sr):
    """librosa로 목표 샘플레이트에 맞춰 wav를 로드해 float 텐서를 반환한다."""
    audio_norm, sampling_rate = librosa.load(full_path, sr=sr, mono=True)
    return torch.FloatTensor(audio_norm.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    """metadata 파일을 읽어 구분자로 각 라인을 분리한다."""
    with open(filename, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text


def get_hparams(init=True):
    """
    CLI 인자를 파싱해 학습 실행용 HParams 객체를 만든다.

    `init=True`일 때 입력 config JSON을 `./logs/<model>/config.json`으로 복사해
    이후 재시작 시 동일 설정을 기준으로 이어서 학습할 수 있게 한다.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/base.json",
        help="JSON file for configuration",
    )
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    parser.add_argument('--port', type=int, default=10000)
    parser.add_argument("-m", "--model", type=str, required=True, help="Model name")
    parser.add_argument('--pretrain_G', type=str, default=None,
                            help='pretrain model')
    parser.add_argument('--pretrain_D', type=str, default=None,
                            help='pretrain model D')
    parser.add_argument('--pretrain_dur', type=str, default=None,
                            help='pretrain model duration')

    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)

    os.makedirs(model_dir, exist_ok=True)

    config_path = args.config
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(config_path, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    hparams.pretrain_G = args.pretrain_G
    hparams.pretrain_D = args.pretrain_D
    hparams.pretrain_dur = args.pretrain_dur
    hparams.port = args.port
    return hparams


def clean_checkpoints(path_to_models="logs/44k/", n_ckpts_to_keep=2, sort_by_time=True):
    """Freeing up space by deleting saved ckpts

    Arguments:
    path_to_models    --  Path to the model directory
    n_ckpts_to_keep   --  Number of ckpts to keep, excluding G_0.pth and D_0.pth
    sort_by_time      --  True -> chronologically delete ckpts
                          False -> lexicographically delete ckpts
    """
    import re

    ckpts_files = [
        f
        for f in os.listdir(path_to_models)
        if os.path.isfile(os.path.join(path_to_models, f))
    ]

    def name_key(_f):
        return int(re.compile("._(\\d+)\\.pth").match(_f).group(1))

    def time_key(_f):
        return os.path.getmtime(os.path.join(path_to_models, _f))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(_x):
        return sorted(
            [f for f in ckpts_files if f.startswith(_x) and not f.endswith("_0.pth")],
            key=sort_key,
        )

    to_del = [
        os.path.join(path_to_models, fn)
        for fn in (x_sorted("G")[:-n_ckpts_to_keep] + x_sorted("D")[:-n_ckpts_to_keep])
    ]

    def del_info(fn):
        return logger.info(f".. Free up space by deleting ckpt {fn}")

    def del_routine(x):
        return [os.remove(x), del_info(x)]

    [del_routine(fn) for fn in to_del]


def get_hparams_from_dir(model_dir):
    """`<model_dir>/config.json`에서 HParams를 로드한다."""
    config_save_path = os.path.join(model_dir, "config.json")
    with open(config_save_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path):
    """config JSON 경로에서 HParams를 로드한다."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = f.read()
    config = json.loads(data)

    hparams = HParams(**config)
    return hparams


def check_git_hash(model_dir):
    """현재 git hash와 모델 디렉토리에 저장된 hash를 비교한다."""
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warn(
            "{} is not a git repository, therefore hash value comparison will be ignored.".format(
                source_dir
            )
        )
        return

    cur_hash = subprocess.getoutput("git rev-parse HEAD")

    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        saved_hash = open(path).read()
        if saved_hash != cur_hash:
            logger.warn(
                "git hash values are different. {}(saved) != {}(current)".format(
                    saved_hash[:8], cur_hash[:8]
                )
            )
    else:
        open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    """모델 디렉토리 기준 파일 로거를 생성해 반환한다."""
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    h = logging.FileHandler(os.path.join(model_dir, filename))
    h.setLevel(logging.DEBUG)
    h.setFormatter(formatter)
    logger.addHandler(h)
    return logger


class HParams:
    """중첩 config를 속성 접근으로 사용할 수 있는 dict 유사 객체."""

    def __init__(self, **kwargs):
        """중첩 dict 값을 재귀적으로 HParams 객체로 감싼다."""
        for k, v in kwargs.items():
            if type(v) == dict:
                v = HParams(**v)
            self[k] = v

    def keys(self):
        """최상위 키 목록을 반환한다."""
        return self.__dict__.keys()

    def items(self):
        """`(key, value)` 쌍을 반환한다."""
        return self.__dict__.items()

    def values(self):
        """최상위 값 목록을 반환한다."""
        return self.__dict__.values()

    def __len__(self):
        """최상위 키 개수를 반환한다."""
        return len(self.__dict__)

    def __getitem__(self, key):
        """dict 스타일 접근을 지원한다. 예: `hps['train']`."""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """dict 스타일 할당을 지원한다."""
        return setattr(self, key, value)

    def __contains__(self, key):
        """`in` 포함 여부 검사를 지원한다."""
        return key in self.__dict__

    def __repr__(self):
        """내부 dictionary의 repr 문자열을 반환한다."""
        return self.__dict__.__repr__()

    def get(self, key, default=None):
        """키가 있으면 값을, 없으면 기본값을 반환한다."""
        return getattr(self, key, default)
