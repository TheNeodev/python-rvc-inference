#!/usr/bin/env python
"""
eiser – A refined voice conversion module for PyPI

This module provides functions to download models, configure the environment,
load voice conversion (VC) models, and perform inference on audio files.
It defines a base configuration class (BaseEiser) that you can use as the
core configuration object when initializing the package.
"""

import os
import sys
import shutil
import gc
import time
import glob
import traceback
import logging
from pathlib import Path
from multiprocessing import cpu_count
from urllib.parse import urlparse
from io import BytesIO
import torch
import numpy as np
import soundfile as sf
import torchaudio
from torchaudio.pipelines import HUBERT_BASE

# Third-party package imports from rvc_inferpy
from rvc_inferpy.split_audio import (
    split_silence_nonsilent,
    adjust_audio_lengths,
    combine_silence_nonsilent,
)
from rvc_inferpy.infer_list.audio import load_audio, wav2
from rvc_inferpy.infer_list.packs.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from rvc_inferpy.pipeline import Pipeline

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# Constants
BASE_DOWNLOAD_LINK = "https://huggingface.co/theNeofr/rvc-base/resolve/main"
BASE_MODELS = ["hubert_base.pt", "rmvpe.pt", "fcpe.pt"]
BASE_DIR = "."


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def load_file_from_url(
    url: str,
    model_dir: str,
    file_name: str | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> str:
    """
    Download a file from the given URL into model_dir.
    Uses a cached file if available.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    if os.path.exists(cached_file):
        if overwrite or os.path.getsize(cached_file) == 0:
            os.remove(cached_file)

    if not os.path.exists(cached_file):
        logger.info(f'Downloading "{url}" to {cached_file}')
        from torch.hub import download_url_to_file

        download_url_to_file(url, cached_file, progress=progress)
    else:
        logger.debug(f"Using cached file: {cached_file}")

    return cached_file


def friendly_name(file: str) -> tuple:
    """
    Extract a friendly model name and extension from a file or URL.
    """
    if file.startswith("http"):
        file = urlparse(file).path
    base = os.path.basename(file)
    model_name, ext = os.path.splitext(base)
    return model_name, ext


def download_manager(
    url: str,
    path: str,
    extension: str = "",
    overwrite: bool = False,
    progress: bool = True,
) -> str:
    """
    Manage download and naming of a file.
    """
    url = url.strip()
    name, ext = friendly_name(url)
    if extension:
        name += f".{extension}"
    else:
        name += ext
    if url.startswith("http"):
        filename = load_file_from_url(
            url=url,
            model_dir=path,
            file_name=name,
            overwrite=overwrite,
            progress=progress,
        )
    else:
        filename = path
    return filename


def note_to_hz(note_name: str) -> float | None:
    """
    Convert a musical note (e.g., 'A4') to its frequency in Hz.
    """
    SEMITONES = {
        "C": -9,
        "C#": -8,
        "D": -7,
        "D#": -6,
        "E": -5,
        "F": -4,
        "F#": -3,
        "G": -2,
        "G#": -1,
        "A": 0,
        "A#": 1,
        "B": 2,
    }
    try:
        pitch_class = note_name[:-1]
        octave = int(note_name[-1])
        semitone = SEMITONES[pitch_class]
        note_number = 12 * (octave - 4) + semitone
        frequency = 440.0 * (2.0 ** (note_number / 12))
        return frequency
    except Exception as e:
        logger.warning(f"Failed to convert note '{note_name}' to Hz: {e}")
        return None





def load_hubert(config, hubert_path: str = None):
    """
    Load and return the HuBERT model using torchaudio.
    If the specified hubert_path does not exist, use the pre-trained torchaudio model.
    """
    if hubert_path is None or not os.path.exists(hubert_path):
        print("Using torchaudio's pre-trained HuBERT model.")
        hubert_model = HUBERT_BASE.get_model()
    else:
        print(f"Loading HuBERT model from {hubert_path}")
        hubert_model = torch.jit.load(hubert_path)

    hubert_model = hubert_model.to(config.device)
    hubert_model = hubert_model.half() if config.is_half else hubert_model.float()
    hubert_model.eval()
    
    return hubert_model


# -----------------------------------------------------------------------------
# Base Configuration Class
# -----------------------------------------------------------------------------
class BaseEiser:
    """
    Base configuration class for Eiser.

    This class encapsulates device configuration and other environment settings.
    It is intended as the base class for initializing your package configuration.
    """

    def __init__(self, device: str = "cuda:0", is_half: bool = True):
        self.device = device
        self.is_half = is_half
        self.n_cpu = cpu_count()
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
        elif torch.backends.mps.is_available():
            logger.info("MPS backend in use for inference.")
            self.device = "mps"
        else:
            logger.info("No GPU found; using CPU for inference.")
            self.device = "cpu"

        # Adjust memory configuration based on precision
        if self.is_half:
            x_pad, x_query, x_center, x_max = 3, 10, 60, 65
        else:
            x_pad, x_query, x_center, x_max = 1, 6, 38, 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = 1, 5, 30, 32

        return x_pad, x_query, x_center, x_max


# -----------------------------------------------------------------------------
# Voice Conversion (VC) Class
# -----------------------------------------------------------------------------
class VC:
    """
    Voice Conversion (VC) class to handle model loading and inference.
    """

    def __init__(self, config: BaseEiser):
        self.config = config
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline = None
        self.cpt = None
        self.version = None
        self.if_f0 = None
        self.hubert_model = None

    def get_vc(self, sid: str, *to_return_protect):
        """
        Load VC model from checkpoint (sid).

        Returns UI update components based on model state.
        If sid is empty, the model cache is cleared.
        """
        logger.info("Loading model checkpoint: " + sid)

        if not sid:
            if self.hubert_model is not None:
                logger.info("Clearing model cache.")
                self._clear_model_cache()
            return (
                {"visible": False, "__type__": "update"},
                {
                    "visible": True,
                    "value": to_return_protect[0] if to_return_protect else 0.5,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": (
                        to_return_protect[1] if len(to_return_protect) > 1 else 0.33
                    ),
                    "__type__": "update",
                },
                "",
                "",
            )
        try:
            logger.info("Loading checkpoint from file.")
            self.cpt = torch.load(sid, map_location="cpu")
        except Exception as e:
            logger.error("Failed to load checkpoint: " + str(e))
            return ({"visible": False, "__type__": "update"}, "", "", "", "")

        self.tgt_sr = self.cpt["config"][-1]
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[
            0
        ]  # Number of speakers
        self.if_f0 = self.cpt.get("f0", 1)
        self.version = self.cpt.get("version", "v1")

        synthesizer_class = {
            ("v1", 1): SynthesizerTrnMs256NSFsid,
            ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
            ("v2", 1): SynthesizerTrnMs768NSFsid,
            ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
        }
        synth_class = synthesizer_class.get(
            (self.version, self.if_f0), SynthesizerTrnMs256NSFsid
        )
        self.net_g = synth_class(*self.cpt["config"], is_half=self.config.is_half)
        del self.net_g.enc_q
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        self.net_g = self.net_g.half() if self.config.is_half else self.net_g.float()
        self.pipeline = Pipeline(self.tgt_sr, self.config)
        n_spk = self.cpt["config"][-3]
        update_info = {"visible": False, "maximum": n_spk, "__type__": "update"}
        if to_return_protect:
            update_info = (
                update_info,
                {
                    "visible": True,
                    "value": to_return_protect[0] if to_return_protect else 0.5,
                    "__type__": "update",
                },
                {
                    "visible": True,
                    "value": (
                        to_return_protect[1] if len(to_return_protect) > 1 else 0.33
                    ),
                    "__type__": "update",
                },
            )
        return update_info

    def _clear_model_cache(self):
        """Clear the model cache."""
        for attr in ["net_g", "n_spk", "hubert_model", "tgt_sr"]:
            setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _parse_pitch(self, pitch, default):
        """Parse a pitch value (digit or note) to a float frequency."""
        if isinstance(pitch, str) and not pitch.isdigit():
            converted = note_to_hz(pitch)
            if converted is not None:
                logger.info(f"Converted pitch '{pitch}' to {converted} Hz.")
                return converted
            else:
                logger.warning(
                    f"Invalid pitch note '{pitch}'. Defaulting to {default} Hz."
                )
                return default
        try:
            return float(pitch)
        except Exception:
            return default

    def _load_audio_and_normalize(self, audio_path, do_formant, quefrency, timbre):
        """Load an audio file and normalize its amplitude."""
        logger.info(f"Loading audio from {audio_path}.")
        audio = load_audio(
            file=audio_path,
            sr=16000,
            DoFormant=do_formant,
            Quefrency=quefrency,
            Timbre=timbre,
        )
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio = audio / audio_max
        return audio

    def _run_inference(
        self,
        sid,
        audio,
        input_audio_path,
        times,
        f0_up_key,
        f0_method,
        file_index,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        crepe_hop_length,
        f0_autotune,
        f0_min,
        f0_max,
    ):
        """Run the inference pipeline and return the output audio."""
        return self.pipeline.pipeline(
            self.hubert_model,
            self.net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            index_rate,
            self.if_f0,
            filter_radius,
            self.tgt_sr,
            resample_sr,
            rms_mix_rate,
            self.version,
            protect,
            crepe_hop_length,
            f0_autotune,
            f0_min=f0_min,
            f0_max=f0_max,
        )

    def vc_single(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        output_format,
        crepe_hop_length,
        do_formant,
        quefrency,
        timbre,
        f0_min,
        f0_max,
        f0_autotune,
        hubert_model_path="hubert_base.pt",
    ):
        """
        Perform inference and save the output audio.

        Returns a tuple:
          (status_info, (sample_rate, audio_data), output_path)
        """
        start_time = time.time()
        if not input_audio_path or not os.path.exists(input_audio_path):
            return "Audio not found.", None, None

        f0_up_key = int(f0_up_key)
        f0_min_val = self._parse_pitch(f0_min, 50)
        f0_max_val = self._parse_pitch(f0_max, 1100)

        try:
            audio = self._load_audio_and_normalize(
                input_audio_path, do_formant, quefrency, timbre
            )
            times = [0, 0, 0]
            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config, hubert_model_path)
            try:
                self.if_f0 = self.cpt.get("f0", 1)
            except Exception:
                msg = "Model was not properly selected."
                logger.error(msg)
                return msg, None, None

            file_index_final = (
                file_index.strip().replace("trained", "added")
                if file_index and isinstance(file_index, str)
                else file_index2 or ""
            )
            audio_opt = self._run_inference(
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index_final,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                crepe_hop_length,
                f0_autotune,
                f0_min_val,
                f0_max_val,
            )
            tgt_sr = (
                resample_sr
                if (self.tgt_sr != resample_sr and resample_sr >= 16000)
                else self.tgt_sr
            )
            index_info = (
                f"Index: {file_index_final}."
                if file_index_final and os.path.exists(file_index_final)
                else "Index not used."
            )
            times.append(time.time() - start_time)

            # Save output audio
            output_dir = os.path.join(os.getcwd(), "output")
            os.makedirs(output_dir, exist_ok=True)
            output_count = 1
            base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
            while True:
                out_filename = f"{base_name}{os.path.basename(os.path.dirname(file_index_final)) if file_index_final else ''}{f0_method.capitalize()}_{output_count}.{output_format}"
                current_output_path = os.path.join(output_dir, out_filename)
                if not os.path.exists(current_output_path):
                    break
                output_count += 1

            try:
                if output_format in ["wav", "flac"]:
                    sf.write(current_output_path, audio_opt, self.tgt_sr)
                else:
                    with BytesIO() as wav_buffer:
                        sf.write(wav_buffer, audio_opt, self.tgt_sr, format="wav")
                        wav_buffer.seek(0)
                        with open(current_output_path, "wb") as outf:
                            wav2(wav_buffer, outf, output_format)
            except Exception as e:
                logger.error("Error saving audio: " + str(e))

            times.append(time.time() - start_time)
            return (
                ("Success.", index_info, times),
                (tgt_sr, audio_opt),
                current_output_path,
            )
        except Exception as e:
            info = traceback.format_exc()
            logger.warning(info)
            return ((str(e), None, [None, None, None, None]), (None, None), None)

    def vc_single_dont_save(
        self,
        sid,
        input_audio_path,
        f0_up_key,
        f0_method,
        file_index,
        file_index2,
        index_rate,
        filter_radius,
        resample_sr,
        rms_mix_rate,
        protect,
        crepe_hop_length,
        do_formant,
        quefrency,
        timbre,
        f0_min,
        f0_max,
        f0_autotune,
        hubert_model_path="hubert_base.pt",
    ):
        """
        Perform inference without saving output audio.

        Returns a tuple:
          (status_info, (sample_rate, audio_data))
        """
        start_time = time.time()
        if not input_audio_path or not os.path.exists(input_audio_path):
            return "Audio not found.", None

        f0_up_key = int(f0_up_key)
        f0_min_val = self._parse_pitch(f0_min, 50)
        f0_max_val = self._parse_pitch(f0_max, 1100)

        try:
            audio = self._load_audio_and_normalize(
                input_audio_path, do_formant, quefrency, timbre
            )
            times = [0, 0, 0]
            if self.hubert_model is None:
                self.hubert_model = load_hubert(self.config, hubert_model_path)
            try:
                self.if_f0 = self.cpt.get("f0", 1)
            except Exception:
                msg = "Model was not properly selected."
                logger.error(msg)
                return msg, None

            file_index_final = (
                file_index.strip().replace("trained", "added")
                if file_index and isinstance(file_index, str)
                else file_index2 or ""
            )
            audio_opt = self._run_inference(
                sid,
                audio,
                input_audio_path,
                times,
                f0_up_key,
                f0_method,
                file_index_final,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                crepe_hop_length,
                f0_autotune,
                f0_min_val,
                f0_max_val,
            )
            tgt_sr = (
                resample_sr
                if (self.tgt_sr != resample_sr and resample_sr >= 16000)
                else self.tgt_sr
            )
            index_info = (
                f"Index: {file_index_final}."
                if file_index_final and os.path.exists(file_index_final)
                else "Index not used."
            )
            times.append(time.time() - start_time)
            return ("Success.", index_info, times), (tgt_sr, audio_opt)
        except Exception as e:
            info = traceback.format_exc()
            logger.warning(info)
            return (str(e), None, [None, None, None, None]), (None, None)


# -----------------------------------------------------------------------------
# Utility Functions
# -----------------------------------------------------------------------------
def get_model(voice_model: str) -> tuple:
    """
    Retrieve the model (.pth) and optional index file from the models directory.
    """
    model_dir = os.path.join(os.getcwd(), "models", voice_model)
    model_filename, index_filename = None, None
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == ".pth":
            model_filename = file
        elif ext == ".index":
            index_filename = file

    if model_filename is None:
        logger.error(f"No model file exists in {model_dir}.")
        return None, None

    return os.path.join(model_dir, model_filename), (
        os.path.join(model_dir, index_filename) if index_filename else ""
    )


# -----------------------------------------------------------------------------
# Main Inference Function
# -----------------------------------------------------------------------------
BASE_DIR = Path(os.getcwd())
sys.path.append(str(BASE_DIR))


def infer_audio(
    model_name: str,
    audio_path: str,
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
    """
    Main function to perform voice conversion inference on an audio file.

    Optionally splits the audio into segments (using silence detection), processes each,
    and then combines the results.
    """
    os.environ["rmvpe_model_path"] = rmvpe_model_path
    os.environ["fcpe_model_path"] = fcpe_model_path
    config = BaseEiser("cuda:0", True)
    vc = VC(config)
    pth_path, index_path = get_model(model_name)
    vc_data = vc.get_vc(pth_path, protect, 0.5)

    if split_infer:
        inferred_files = []
        temp_dir = os.path.join(os.getcwd(), "seperate", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        logger.info("Splitting audio into silence and nonsilent segments.")
        silence_files, nonsilent_files = split_silence_nonsilent(
            audio_path, min_silence, silence_threshold, seek_step, keep_silence
        )
        logger.info(
            f"Silence segments: {len(silence_files)}; Nonsilent segments: {len(nonsilent_files)}."
        )

        for i, nonsilent_file in enumerate(nonsilent_files):
            logger.info(f"Inferring nonsilent segment {i+1}.")
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
                logger.info("Inference successful.")
                logger.info(inference_info[1])
                logger.info(
                    "Times: npy: %.2fs, f0: %.2fs, infer: %.2fs, Total: %.2fs"
                    % tuple(inference_info[2])
                )
            else:
                logger.error("Error during inference: " + str(inference_info[0]))
                return None
            inferred_files.append(output_path)

        logger.info("Adjusting inferred audio lengths.")
        adjusted_inferred_files = adjust_audio_lengths(nonsilent_files, inferred_files)
        output_count = 1
        output_dir = os.path.join(os.getcwd(), "output")
        os.makedirs(output_dir, exist_ok=True)
        while True:
            out_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}{model_name}{f0_method.capitalize()}_{output_count}.{audio_format}"
            output_path = os.path.join(output_dir, out_filename)
            if not os.path.exists(output_path):
                break
            output_count += 1
        output_path = combine_silence_nonsilent(
            silence_files, adjusted_inferred_files, keep_silence, output_path
        )
        for inferred_file in inferred_files:
            shutil.move(inferred_file, temp_dir)
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
            logger.info("Inference successful.")
            logger.info(inference_info[1])
            logger.info(
                "Times: npy: %.2fs, f0: %.2fs, infer: %.2fs, Total: %.2fs"
                % tuple(inference_info[2])
            )
        else:
            logger.error("Error during inference: " + str(inference_info[0]))
            del config, vc
            gc.collect()
            return inference_info[0]

    del config, vc
    gc.collect()
    return output_path
