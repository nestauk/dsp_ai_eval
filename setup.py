"""dsp_ai_eval."""

from pathlib import Path
from setuptools import find_packages
from setuptools import setup


def read_lines(path):
    """Read lines of `path`."""
    with open(path) as f:
        return f.read().splitlines()


BASE_DIR = Path(__file__).parent


setup(
    name="dsp_ai_eval",
    long_description=open(BASE_DIR / "README.md").read(),
    install_requires=read_lines(BASE_DIR / "requirements.txt"),
    extras_require={"dev": read_lines(BASE_DIR / "requirements_dev.txt")},
    packages=find_packages(exclude=["docs"]),
    version="0.1.0",
    description="A rapid exploration of evaluating outputs from generative AI tools",
    author="Nesta",
    license="proprietary",
    entry_points={"console_scripts": ["dsp_ai_eval=dsp_ai_eval.cli:app"]},
)
