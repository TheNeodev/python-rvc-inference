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

not supported

### As a Dependency in a Python Project

You can also use `rvc_inferpy` directly in your Python projects. Here's an example:

# Example usage of the Eiser package
```
from rvc_inferpy import BaseEiser, VC, infer_audio
```
# Initialize the configuration
```
config = BaseEiser(device="cuda:0", is_half=True)
```
# Option 1: Using the VC class directly for more control

```
vc_instance = VC(config)
# Load your model checkpoint (replace 'path/to/model.pth' with your model file path)
update_info = vc_instance.get_vc('path/to/model.pth')
# Perform inference without saving (adjust the parameters as needed)
status_info, audio_result = vc_instance.vc_single_dont_save(
    sid='path/to/model.pth',
    input_audio_path='path/to/input.wav',
    f0_up_key=0,
    f0_method="rmvpe+",
    file_index="",
    file_index2="",
    index_rate=0.75,
    filter_radius=3,
    resample_sr=16000,
    rms_mix_rate=0.25,
    protect=0.33,
    crepe_hop_length=128,
    do_formant=False,
    quefrency=0,
    timbre=1,
    f0_min="50",
    f0_max="1100",
    f0_autotune=False,
)
print(status_info)
```

# Option 2: Using the high-level infer_audio function

```

output_path = infer_audio(
    model_name="your_voice_model",
    audio_path="path/to/input.wav",
    f0_change=0,
    f0_method="rmvpe+",
    min_pitch="50",
    max_pitch="1100",
    crepe_hop_length=128,
    index_rate=0.75,
    filter_radius=3,
    rms_mix_rate=0.25,
    protect=0.33,
    split_infer=False,  # Change to True if you want to split the audio for inference
    do_formant=False,
    quefrency=0,
    timbre=1,
    f0_autotune=False,
    audio_format="wav",
)
print("Inference output saved to:", output_path)
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


