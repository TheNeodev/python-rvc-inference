# python RVC inference

> [!NOTE]
 > This project is still under development.</a>.

 ![PyPI](https://img.shields.io/pypi/v/rvc-inferpy?logo=pypi&logoColor=white)
![GitHub forks](https://img.shields.io/github/forks/TheNeodev/rvc_inferpy?style=flat) [![GitHub Stars](https://img.shields.io/github/stars/TheNeodev/rvc_inferpy?style=flat&logo=github&label=Star&color=blue)](https://github.com/TheNeodev/rvc_inferpy/stargazers)

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
rvc-cli -h
```

Here’s a breakdown of the full command-line options:

```bash
usage: rvc-cli [-h] [--model_name MODEL_NAME] [--audio_path AUDIO_PATH] 
                 [--f0_change F0_CHANGE] [--f0_method F0_METHOD] 
                 [--min_pitch MIN_PITCH] [--max_pitch MAX_PITCH] 
                 [--crepe_hop_length CREPE_HOP_LENGTH] [--index_rate INDEX_RATE] 
                 [--filter_radius FILTER_RADIUS] [--rms_mix_rate RMS_MIX_RATE] 
                 [--protect PROTECT] [--split_infer] [--min_silence MIN_SILENCE] 
                 [--silence_threshold SILENCE_THRESHOLD] [--seek_step SEEK_STEP] 
                 [--keep_silence KEEP_SILENCE] [--do_formant] [--quefrency QUEFRENCY] 
                 [--timbre TIMBRE] [--f0_autotune] [--audio_format AUDIO_FORMAT] 
                 [--resample_sr RESAMPLE_SR] 
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


### Example Command:

```bash
rvc-cli --model_name "model_name_here" --audio_path "path_to_audio.wav" --f0_change 0 --f0_method "crepe" --min_pitch 50 --max_pitch 800
```

### As a Dependency in a Python Project

You can also use `rvc_inferpy` directly in your Python projects. Here's an example:

```python
from rvc_inferpy import infer_audio

inferred_audio = infer_audio(
    model_name="model_name_here",       # Name or path to the RVC model
    audio_path="path_to_audio.wav",     # Path to the input audio file
    f0_change=0,                        # Change in fundamental frequency
    f0_method="crepe",                  # F0 extraction method ("crepe", "dio", etc.) 
    crepe_hop_length=128,               # Hop length for Crepe
    index_rate=1.0,                     # Index rate for model inference
    filter_radius=3,                    # Radius for smoothing filters
    rms_mix_rate=0.75,                  # Mixing rate for RMS
    protect=0.33,                       # Protect level to prevent overfitting
    split_infer=True,                   # Whether to split audio for inference
    min_silence=0.5,                    # Minimum silence duration for splitting
    silence_threshold=-40,              # Silence threshold in dB
    seek_step=10,                       # Seek step in milliseconds
    keep_silence=0.1,                   # Keep silence duration in seconds
    quefrency=0.0,                      # Cepstrum quefrency adjustment
    tumbre=1.0,                         # Timbre preservation level
    f0_autotune=False,                  # Enable or disable F0 autotuning
    
)
```

The `infer_audio` function will return the processed audio object based on the provided parameters




> [!TIP]
 > Ensure that you upload your models in the `models/{model_name}` folder.</a>





## Terms of Use

The use of the converted voice for the following purposes is prohibited.

* Criticizing or attacking individuals.

* Advocating for or opposing specific political positions, religions, or ideologies.

* Publicly displaying strongly stimulating expressions without proper zoning.

* Selling of voice models and generated voice clips.

* Impersonation of the original owner of the voice with malicious intentions to harm/hurt others.

* Fraudulent purposes that lead to identity theft or fraudulent phone calls.

## Disclaimer

I am not liable for any direct, indirect, consequential, incidental, or special damages arising out of or in any way connected with the use/misuse or inability to use this software.



## Credits

- **IAHispano's Applio**: Base of this project.
- **RVC-Project**: Original RVC repository.

## License

This project is licensed under the [MIT License]((https://github.com/TheNeodev/rvc_inferpy/tree/main?tab=MIT-1-ov-file)]).


