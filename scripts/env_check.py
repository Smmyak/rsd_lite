from __future__ import annotations

import importlib.util
import platform
import shutil


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> None:
    print(f"python: {platform.python_version()}")
    for name in ["torch", "torchvision", "PIL", "numpy", "pytest", "kaggle", "mamba_ssm"]:
        print(f"{name}: {'yes' if has_module(name) else 'no'}")
    print(f"kaggle cli: {'yes' if shutil.which('kaggle') else 'no'}")


if __name__ == "__main__":
    main()

