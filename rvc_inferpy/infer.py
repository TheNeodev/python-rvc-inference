import os, sys
import shutil
import gc
import torch
from multiprocessing import cpu_count
from rvc_inferpy.modules import VC
from rvc_inferpy.split_audio import (
    split_silence_nonsilent,
    adjust_audio_lengths,
    combine_silence_nonsilent,
)
from pathlib import Path
import requests


class Configs:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = 0
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
        elif torch.backends.mps.is_available():
            print("No supported N-card found, use MPS for inference")
            self.device = "mps"
        else:
            print("No supported N-card found, use CPU for inference")
            self.device = "cpu"

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G memory config
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G memory config
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem != None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32

        return x_pad, x_query, x_center, x_max


def get_model(voice_model):
    model_dir = os.path.join(os.getcwd(), "models", voice_model)
    model_filename, index_filename = None, None
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == ".pth":
            model_filename = file
        if ext == ".index":
            index_filename = file

    if model_filename is None:
        print(f"No model file exists in {models_dir}.")
        return None, None

    return os.path.join(model_dir, model_filename), (
        os.path.join(model_dir, index_filename) if index_filename else ""
    )


BASE_DIR = Path(os.getcwd())
sys.path.append(str(BASE_DIR))

files_to_check = ["hubert_base.pt", "rmvpe.pt", "fcpe.pt"]

missing_files = [file for file in files_to_check if not (BASE_DIR / file).exists()]


def dl_model(link, model_name, dir_name):
    url = f"{link}/{model_name}"
    response = requests.get(url, stream=True)
    response.raise_for_status()

    target_path = dir_name / model_name
    target_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Create the directory if it doesn't exist

    with open(target_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"{model_name} downloaded successfully!")


if missing_files:
    RVC_DOWNLOAD_LINK = "https://huggingface.co/theNeofr/rvc-base/resolve/main"  # Replace with the actual download link

    for model in missing_files:
        print(f"Downloading {model}...")
        dl_model(RVC_DOWNLOAD_LINK, model, BASE_DIR)

    print("All missing models have been downloaded!")
else:
    pass


def infer_audio(
    model_name,
    audio_path,
    f0_change=0,
    f0_method="rmvpe+",
    min_pitch="50",
    max_pitch="1100",
    crepe_hop_length=128,
    index_rate=0.75,
    filter_radius=3,
    rms_mix_rate=0.25,
    protect=0.33,
    split_infer=False,
    min_silence=500,
    silence_threshold=-50,
    seek_step=1,
    keep_silence=100,
    do_formant=False,
    quefrency=0,
    timbre=1,
    f0_autotune=False,
    audio_format="wav",
    resample_sr=0,
    hubert_model_path="hubert_base.pt",
    rmvpe_model_path="rmvpe.pt",
    fcpe_model_path="fcpe.pt",
):
    os.environ["rmvpe_model_path"] = rmvpe_model_path
    os.environ["fcpe_model_path"] = fcpe_model_path
    configs = Configs("cuda:0", True)
    vc = VC(configs)
    pth_path, index_path = get_model(model_name)
    vc_data = vc.get_vc(pth_path, protect, 0.5)

    if split_infer:
        inferred_files = []
        temp_dir = os.path.join(os.getcwd(), "seperate", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        print("Splitting audio to silence and nonsilent segments.")
        silence_files, nonsilent_files = split_silence_nonsilent(
            audio_path, min_silence, silence_threshold, seek_step, keep_silence
        )
        print(
            f"Total silence segments: {len(silence_files)}.\nTotal nonsilent segments: {len(nonsilent_files)}."
        )
        for i, nonsilent_file in enumerate(nonsilent_files):
            print(f"Inferring nonsilent audio {i+1}")
            inference_info, audio_data, output_path = vc.vc_single(
                0,
                nonsilent_file,
                f0_change,
                f0_method,
                index_path,
                index_path,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                audio_format,
                crepe_hop_length,
                do_formant,
                quefrency,
                timbre,
                min_pitch,
                max_pitch,
                f0_autotune,
                hubert_model_path,
            )
            if inference_info[0] == "Success.":
                print("Inference ran successfully.")
                print(inference_info[1])
                print(
                    "Times:\nnpy: %.2fs f0: %.2fs infer: %.2fs\nTotal time: %.2fs"
                    % (*inference_info[2],)
                )
            else:
                print(f"An error occurred while processing.\n{inference_info[0]}")
                return None
            inferred_files.append(output_path)
        print("Adjusting inferred audio lengths.")
        adjusted_inferred_files = adjust_audio_lengths(nonsilent_files, inferred_files)
        print("Combining silence and inferred audios.")
        output_count = 1
        while True:
            output_path = os.path.join(
                os.getcwd(),
                "output",
                f"{os.path.splitext(os.path.basename(audio_path))[0]}{model_name}{f0_method.capitalize()}_{output_count}.{audio_format}",
            )
            if not os.path.exists(output_path):
                break
            output_count += 1
        output_path = combine_silence_nonsilent(
            silence_files, adjusted_inferred_files, keep_silence, output_path
        )
        [shutil.move(inferred_file, temp_dir) for inferred_file in inferred_files]
        shutil.rmtree(temp_dir)
    else:
        inference_info, audio_data, output_path = vc.vc_single(
            0,
            audio_path,
            f0_change,
            f0_method,
            index_path,
            index_path,
            index_rate,
            filter_radius,
            resample_sr,
            rms_mix_rate,
            protect,
            audio_format,
            crepe_hop_length,
            do_formant,
            quefrency,
            timbre,
            min_pitch,
            max_pitch,
            f0_autotune,
            hubert_model_path,
        )
        if inference_info[0] == "Success.":
            print("Inference ran successfully.")
            print(inference_info[1])
            print(
                "Times:\nnpy: %.2fs f0: %.2fs infer: %.2fs\nTotal time: %.2fs"
                % (*inference_info[2],)
            )
        else:
            print(f"An error occurred while processing.\n{inference_info[0]}")
            del configs, vc
            gc.collect()
            return inference_info[0]

    del configs, vc
    gc.collect()
    return output_path


class infernew:
    def __init__(
        self,
        model_name,
        sound_path,
        f0_change=0,
        f0_method="rmvpe",
        min_pitch=50,
        max_pitch=800,
        crepe_hop_length=128,
        index_rate=1.0,
        filter_radius=3,
        rms_mix_rate=0.75,
        protect=0.33,
        split_infer=True,
        min_silence=0.5,
        silence_threshold=-40,
        seek_step=10,
        keep_silence=0.1,
        quefrency=0.0,
        timbre=1.0,
        f0_autotune=False,
        output_format="wav",
    ):
        self.model_name = model_name
        self.sound_path = sound_path
        self.f0_change = f0_change
        self.f0_method = f0_method
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch
        self.crepe_hop_length = crepe_hop_length
        self.index_rate = index_rate
        self.filter_radius = filter_radius
        self.rms_mix_rate = rms_mix_rate
        self.protect = protect
        self.split_infer = split_infer
        self.min_silence = min_silence
        self.silence_threshold = silence_threshold
        self.seek_step = seek_step
        self.keep_silence = keep_silence
        self.quefrency = quefrency
        self.timbre = timbre
        self.f0_autotune = f0_autotune
        self.output_format = output_format

    def run_inference(self):
        inferred_audio = infer_audio(
            MODEL_NAME=self.model_name,
            SOUND_PATH=self.sound_path,
            F0_CHANGE=self.f0_change,
            F0_METHOD=self.f0_method,
            MIN_PITCH=self.min_pitch,
            MAX_PITCH=self.max_pitch,
            CREPE_HOP_LENGTH=self.crepe_hop_length,
            INDEX_RATE=self.index_rate,
            FILTER_RADIUS=self.filter_radius,
            RMS_MIX_RATE=self.rms_mix_rate,
            PROTECT=self.protect,
            SPLIT_INFER=self.split_infer,
            MIN_SILENCE=self.min_silence,
            SILENCE_THRESHOLD=self.silence_threshold,
            SEEK_STEP=self.seek_step,
            KEEP_SILENCE=self.keep_silence,
            QUEFRENCY=self.quefrency,
            TIMBRE=self.timbre,
            F0_AUTOTUNE=self.f0_autotune,
            OUTPUT_FORMAT=self.output_format,
        )
        return inferred_audio
