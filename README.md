# RVC INFER PY 

[![PyPI version](https://badge.fury.io/py/rvc-inferpy.svg)](https://badge.fury.io/py/rvc-inferpy)


`rvc_inferpy` is a Python library designed for audio inference using RVC (Retrieval-based Voice Conversion). It provides a straightforward interface to process audio files with various configurable parameters.

## Installation

Install the package using pip:

```bash
pip install rvc-inferpy
```

## Usage

Here's a simple example demonstrating how to use the library:

```python
from rvc_inferpy import infer_audio

inferred_audio = infer_audio(
    MODEL_NAME="model_name_here",       # Name or path to the RVC model
    SOUND_PATH="path_to_audio.wav",    # Path to the input audio file
    F0_CHANGE=0,                       # Change in fundamental frequency
    F0_METHOD="crepe",                 # F0 extraction method ("crepe", "dio", etc.)
    MIN_PITCH=50,                      # Minimum pitch value
    MAX_PITCH=800,                     # Maximum pitch value
    CREPE_HOP_LENGTH=128,              # Hop length for Crepe
    INDEX_RATE=1.0,                    # Index rate for model inference
    FILTER_RADIUS=3,                   # Radius for smoothing filters
    RMS_MIX_RATE=0.75,                 # Mixing rate for RMS
    PROTECT=0.33,                      # Protect level to prevent overfitting
    SPLIT_INFER=True,                  # Whether to split audio for inference
    MIN_SILENCE=0.5,                   # Minimum silence duration for splitting
    SILENCE_THRESHOLD=-40,             # Silence threshold in dB
    SEEK_STEP=10,                      # Seek step in milliseconds
    KEEP_SILENCE=0.1,                  # Keep silence duration in seconds
    FORMANT_SHIFT=0.0,                 # Amount of formant shifting
    QUEFRENCY=0.0,                     # Cepstrum quefrency adjustment
    TIMBRE=1.0,                        # Timbre preservation level
    F0_AUTOTUNE=False,                 # Enable or disable F0 autotuning
    OUTPUT_FORMAT="wav"                # Desired output format (e.g., "wav", "mp3")
)
```
## Usage with cli

you can also use with cli by:

```

rvc-infer -h


```


## Parameters

- **`MODEL_NAME`**: Name or path of the RVC model to use.
- **`SOUND_PATH`**: Path to the input audio file to be processed.
- **`F0_CHANGE`**: Adjusts the fundamental frequency (F0) of the audio.
- **`F0_METHOD`**: Method for extracting F0 (e.g., `"crepe"`, `"dio"`).
- **`MIN_PITCH`** / **`MAX_PITCH`**: Minimum and maximum pitch values for processing.
- **`CREPE_HOP_LENGTH`**: Hop length parameter for the Crepe method.
- **`INDEX_RATE`**: Determines the index rate for the inference model.
- **`FILTER_RADIUS`**: Radius used for smoothing filters.
- **`RMS_MIX_RATE`**: Mix rate for RMS adjustments.
- **`PROTECT`**: Protects specific audio characteristics from overfitting.
- **`SPLIT_INFER`**: Splits the audio for inference if set to `True`.
- **`MIN_SILENCE`**: Minimum silence duration for splitting audio (in seconds).
- **`SILENCE_THRESHOLD`**: Threshold to detect silence (in decibels).
- **`SEEK_STEP`**: Seek step in milliseconds during splitting.
- **`KEEP_SILENCE`**: Duration of silence to retain after processing.
- **`FORMANT_SHIFT`**: Amount of formant shifting applied.
- **`QUEFRENCY`**: Adjusts the quefrency in the cepstrum domain.
- **`TIMBRE`**: Controls timbre preservation during processing.
- **`F0_AUTOTUNE`**: Enables or disables F0 autotuning.
- **`OUTPUT_FORMAT`**: Specifies the output file format (e.g., `"wav"`, `"mp3"`).

## Output

The function returns an audio object with the processed audio based on the provided parameters.


## Information

you must upload your models in `models/model_name` folder

## Credits
IAHispano's Applio: base of this project

RVC-Project: Original RVC repository


## License

This project is licensed under the [MIT License](LICENSE).

