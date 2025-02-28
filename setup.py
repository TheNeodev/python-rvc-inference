import setuptools
import os

# Read the long description from README.md
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rvc_inferpy",
    version="0.7.1",
    author="TheNeoDev",
    author_email="theneodevemail@gmail.com",
    description="Easy tools for RVC Inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    packages=setuptools.find_packages(),
    install_requires=[
        "av",
        "ffmpeg-python",
        "faiss-cpu",
        "praat-parselmouth==0.4.2",
        "pyworld==0.3.4",
        "librosa",
        "tqdm",
        "resampy==0.4.2",
        "fairseq==0.12.2",
        "pydub",
        "einops",
        "local_attention",
        "torchcrepe==0.0.23",
        "torchfcpe",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
