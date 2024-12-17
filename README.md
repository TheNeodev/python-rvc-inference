

# RVC INFERPY

> [!NOTE]
 > This project is still under development.</a>. 


[![PyPI version](https://badge.fury.io/py/rvc-inferpy.svg)](https://badge.fury.io/py/rvc-inferpy)

`rvc_inferpy` is a Python library for performing audio inference with RVC (Retrieval-based Voice Conversion). It provides a simple command-line interface (CLI) and can be integrated into Python projects for audio processing with customizable parameters.

## Installation

You can install the package using `pip`:

```bash
pip install rvc-inferpy
```

## Usage

### Command Line Interface (CLI)

You can interact with `rvc_inferpy` through the command line. To view the available options and how to use the tool, run:

```bash
rvc-infer -h
```

Here’s a breakdown of the full command-line options:

```bash
usage: rvc-infer [-h] [--model_name MODEL_NAME] [--audio_path AUDIO_PATH] 
                 [--f0_change F0_CHANGE] [--f0_method F0_METHOD] 
                 [--min_pitch MIN_PITCH] [--max_pitch MAX_PITCH] 
                 [--crepe_hop_length CREPE_HOP_LENGTH] [--index_rate INDEX_RATE] 
                 [--filter_radius FILTER_RADIUS] [--rms_mix_rate RMS_MIX_RATE] 
                 [--protect PROTECT] [--split_infer] [--min_silence MIN_SILENCE] 
                 [--silence_threshold SILENCE_THRESHOLD] [--seek_step SEEK_STEP] 
                 [--keep_silence KEEP_SILENCE] [--do_formant] [--quefrency QUEFRENCY] 
                 [--timbre TIMBRE] [--f0_autotune] [--audio_format AUDIO_FORMAT] 
                 [--resample_sr RESAMPLE_SR] [--hubert_model_path HUBERT_MODEL_PATH] 
                 [--rmvpe_model_path RMVPE_MODEL_PATH] [--fcpe_model_path FCPE_MODEL_PATH]
```

### Command-Line Options:
- `-h, --help`: Show help message and exit.
- `--model_name MODEL_NAME`: Name or path of the model.
- `--audio_path AUDIO_PATH`: Path to the input audio file.
- `--f0_change F0_CHANGE`: Pitch change factor.
- `--f0_method F0_METHOD`: Method for F0 estimation (e.g., "crepe").
- `--min_pitch MIN_PITCH`: Minimum pitch value.
- `--max_pitch MAX_PITCH`: Maximum pitch value.
- `--crepe_hop_length CREPE_HOP_LENGTH`: Crepe hop length.
- `--index_rate INDEX_RATE`: Index rate.
- `--filter_radius FILTER_RADIUS`: Filter radius.
- `--rms_mix_rate RMS_MIX_RATE`: RMS mix rate.
- `--protect PROTECT`: Protect factor to avoid distortion.
- `--split_infer`: Enable split inference.
- `--min_silence MIN_SILENCE`: Minimum silence duration (in seconds).
- `--silence_threshold SILENCE_THRESHOLD`: Silence threshold in dB.
- `--seek_step SEEK_STEP`: Step size for silence detection.
- `--keep_silence KEEP_SILENCE`: Duration to keep silence (in seconds).
- `--do_formant`: Enable formant processing.
- `--quefrency QUEFRENCY`: Quefrency adjustment.
- `--timbre TIMBRE`: Timbre adjustment factor.
- `--f0_autotune`: Enable automatic F0 tuning.
- `--audio_format AUDIO_FORMAT`: Desired output audio format (e.g., "wav", "mp3").
- `--resample_sr RESAMPLE_SR`: Resample sample rate.
- `--hubert_model_path HUBERT_MODEL_PATH`: Path to Hubert model.
- `--rmvpe_model_path RMVPE_MODEL_PATH`: Path to RMVPE model.
- `--fcpe_model_path FCPE_MODEL_PATH`: Path to FCPE model.

### Example Command:

```bash
rvc-infer --model_name "model_name_here" --audio_path "path_to_audio.wav" --f0_change 0 --f0_method "crepe" --min_pitch 50 --max_pitch 800
```

### As a Dependency in a Python Project

You can also use `rvc_inferpy` directly in your Python projects. Here's an example:

```python
from rvc_inferpy import infer_audio

inferred_audio = infer_audio(
    MODEL_NAME="model_name_here",       # Name or path to the RVC model
    SOUND_PATH="path_to_audio.wav",     # Path to the input audio file
    F0_CHANGE=0,                        # Change in fundamental frequency
    F0_METHOD="crepe",                  # F0 extraction method ("crepe", "dio", etc.)
    MIN_PITCH=50,                       # Minimum pitch value
    MAX_PITCH=800,                      # Maximum pitch value
    CREPE_HOP_LENGTH=128,               # Hop length for Crepe
    INDEX_RATE=1.0,                     # Index rate for model inference
    FILTER_RADIUS=3,                    # Radius for smoothing filters
    RMS_MIX_RATE=0.75,                  # Mixing rate for RMS
    PROTECT=0.33,                       # Protect level to prevent overfitting
    SPLIT_INFER=True,                   # Whether to split audio for inference
    MIN_SILENCE=0.5,                    # Minimum silence duration for splitting
    SILENCE_THRESHOLD=-40,              # Silence threshold in dB
    SEEK_STEP=10,                       # Seek step in milliseconds
    KEEP_SILENCE=0.1,                   # Keep silence duration in seconds
    QUEFRENCY=0.0,                      # Cepstrum quefrency adjustment
    TIMBRE=1.0,                         # Timbre preservation level
    F0_AUTOTUNE=False,                  # Enable or disable F0 autotuning
    OUTPUT_FORMAT="wav"                 # Desired output format (e.g., "wav", "mp3")
)
```

The `infer_audio` function will return the processed audio object based on the provided parameters.

## Model Files

Ensure that you upload your models in the `models/{model_name}` folder.

## Credits

- **IAHispano's Applio**: Base of this project.
- **RVC-Project**: Original RVC repository.

## License

This project is licensed under the [MIT License](https://github.com/TheNeodev/rvc_inferpy/tree/main#).

