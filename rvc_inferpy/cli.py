# dummy cli stuff 😏

import argparse
import os
import shutil
import gc
from rvc_inferpy.modules import VC
from rvc_inferpy.infer import Configs, get_model
from rvc_inferpy.split_audio import (
    split_silence_nonsilent,
    adjust_audio_lengths,
    combine_silence_nonsilent,
)


def infer_audio_cli():
    parser = argparse.ArgumentParser(description="RVC INFERPY CLI VER.")
    parser.add_argument("--model_name", type=str, help="Name of the model.")
    parser.add_argument("--audio_path", type=str, help="Path to the input audio file.")
    parser.add_argument(
        "--f0_change", type=float, default=0, help="Pitch change factor."
    )
    parser.add_argument(
        "--f0_method", type=str, default="rmvpe+", help="Method for F0 estimation."
    )
    parser.add_argument(
        "--min_pitch", type=str, default="50", help="Minimum pitch value."
    )
    parser.add_argument(
        "--max_pitch", type=str, default="1100", help="Maximum pitch value."
    )
    parser.add_argument(
        "--crepe_hop_length", type=int, default=128, help="Crepe hop length."
    )
    parser.add_argument("--index_rate", type=float, default=0.75, help="Index rate.")
    parser.add_argument("--filter_radius", type=int, default=3, help="Filter radius.")
    parser.add_argument(
        "--rms_mix_rate", type=float, default=0.25, help="RMS mix rate."
    )
    parser.add_argument("--protect", type=float, default=0.33, help="Protect factor.")
    parser.add_argument(
        "--split_infer", action="store_true", help="Enable split inference."
    )
    parser.add_argument(
        "--min_silence", type=int, default=500, help="Minimum silence duration."
    )
    parser.add_argument(
        "--silence_threshold", type=float, default=-50, help="Silence threshold (dB)."
    )
    parser.add_argument(
        "--seek_step", type=int, default=1, help="Seek step for silence detection."
    )
    parser.add_argument(
        "--keep_silence", type=int, default=100, help="Silence retention duration."
    )
    parser.add_argument(
        "--do_formant", action="store_true", help="Enable formant processing."
    )
    parser.add_argument(
        "--quefrency", type=float, default=0, help="Quefrency adjustment value."
    )
    parser.add_argument(
        "--timbre", type=float, default=1, help="Timbre adjustment factor."
    )
    parser.add_argument(
        "--f0_autotune", action="store_true", help="Enable F0 autotuning."
    )
    parser.add_argument(
        "--audio_format", type=str, default="wav", help="Output audio format."
    )
    parser.add_argument(
        "--resample_sr", type=int, default=0, help="Resample sample rate."
    )
    parser.add_argument(
        "--hubert_model_path",
        type=str,
        default="hubert_base.pt",
        help="Path to Hubert model.",
    )
    parser.add_argument(
        "--rmvpe_model_path", type=str, default="rmvpe.pt", help="Path to RMVPE model."
    )
    parser.add_argument(
        "--fcpe_model_path", type=str, default="fcpe.pt", help="Path to FCPE model."
    )
    args = parser.parse_args()

    os.environ["rmvpe_model_path"] = args.rmvpe_model_path
    os.environ["fcpe_model_path"] = args.fcpe_model_path
    configs = Configs("cuda:0", True)
    vc = VC(configs)
    pth_path, index_path = get_model(args.model_name)
    vc_data = vc.get_vc(pth_path, args.protect, 0.5)

    if args.split_infer:
        inferred_files = []
        temp_dir = os.path.join(os.getcwd(), "seperate", "temp")
        os.makedirs(temp_dir, exist_ok=True)
        print("Splitting audio into silence and nonsilent segments.")
        silence_files, nonsilent_files = split_silence_nonsilent(
            args.audio_path,
            args.min_silence,
            args.silence_threshold,
            args.seek_step,
            args.keep_silence,
        )
        for i, nonsilent_file in enumerate(nonsilent_files):
            print(f"Processing nonsilent audio {i+1}/{len(nonsilent_files)}")
            inference_info, audio_data, output_path = vc.vc_single(
                0,
                nonsilent_file,
                args.f0_change,
                args.f0_method,
                index_path,
                index_path,
                args.index_rate,
                args.filter_radius,
                args.resample_sr,
                args.rms_mix_rate,
                args.protect,
                args.audio_format,
                args.crepe_hop_length,
                args.do_formant,
                args.quefrency,
                args.timbre,
                args.min_pitch,
                args.max_pitch,
                args.f0_autotune,
                args.hubert_model_path,
            )
            if inference_info[0] == "Success.":
                print("Inference ran successfully.")
                print(inference_info[1])
            else:
                print(f"Error: {inference_info[0]}")
                return
            inferred_files.append(output_path)

        adjusted_inferred_files = adjust_audio_lengths(nonsilent_files, inferred_files)
        output_path = combine_silence_nonsilent(
            silence_files, adjusted_inferred_files, args.keep_silence, output_path
        )
        shutil.rmtree(temp_dir)
    else:
        inference_info, audio_data, output_path = vc.vc_single(
            0,
            args.audio_path,
            args.f0_change,
            args.f0_method,
            index_path,
            index_path,
            args.index_rate,
            args.filter_radius,
            args.resample_sr,
            args.rms_mix_rate,
            args.protect,
            args.audio_format,
            args.crepe_hop_length,
            args.do_formant,
            args.quefrency,
            args.timbre,
            args.min_pitch,
            args.max_pitch,
            args.f0_autotune,
            args.hubert_model_path,
        )
        if inference_info[0] == "Success.":
            print("Inference ran successfully.")
            print(inference_info[1])
        else:
            print(f"Error: {inference_info[0]}")
            return

    del configs, vc
    gc.collect()
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    infer_audio_cli()
