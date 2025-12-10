"""Setup script for Local LLMs package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="local-llms",
    version="0.1.0",
    author="lostinmath",
    description="Privacy-focused, efficient custom language models for personal use",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/lostinmath/local_llms",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.35.0",
        "torch>=2.0.0",
        "sentencepiece>=0.1.99",
        "accelerate>=0.24.0",
        "bitsandbytes>=0.41.0",
        "optimum>=1.14.0",
        "cryptography>=41.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "peft>=0.6.0",
        "datasets>=2.14.0",
        "trl>=0.7.4",
        "pyyaml>=6.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.66.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    # Note: CLI entry point can be added when CLI module is implemented
    # entry_points={
    #     "console_scripts": [
    #         "local-llms=local_llms.cli:main",
    #     ],
    # },
    include_package_data=True,
    zip_safe=False,
)
