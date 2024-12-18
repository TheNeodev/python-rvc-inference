import os
from pathlib import Path

# Configuration: Environment variables for model paths
hubert_model_path = os.environ.get("hubert_model_path")
rmvpe_model_path = os.environ.get("rmvpe_model_path")
fcpe_model_path = os.environ.get("fcpe_model_path")


# Base Loader
def BaseLoader(hubert_path=None, rmvpe_path=None):
    print(
        f"Initialized BaseLoader with:\nHubert Path: {hubert_path}\nRMVPE Path: {rmvpe_path}"
    )


# Check for required files or environment configurations
BASE_DIR = Path(".")
required_files = {
    "hubert_base.pt": hubert_model_path,
    "rmvpe.pt": rmvpe_model_path,
    "fcpe.pt": fcpe_model_path,
}


def validate_config_and_files():
    """Validates that required model configurations or files are present."""
    missing_configs = {
        file: env
        for file, env in required_files.items()
        if env is None and not (BASE_DIR / file).exists()
    }

    if missing_configs:
        print("Missing required configurations or files:")
        for file, env in missing_configs.items():
            print(f"- {file}: Config not found in environment or local directory.")
        raise RuntimeError(
            "Ensure all model paths are correctly set in environment variables or files exist in the directory."
        )
    else:
        print("All configurations and required files are accounted for.")
