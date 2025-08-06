#!/usr/bin/env python3
"""
ORION Platform Setup Script
==========================

Setup script for the ORION materials science platform.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = []
if (this_directory / "requirements.txt").exists():
    requirements = (this_directory / "requirements.txt").read_text().splitlines()

setup(
    name="orion-platform",
    version="1.0.0",
    author="ORION Development Team",
    author_email="support@orion-materials.ai",
    description="ORION: Optimized Research & Innovation for Organized Nanomaterials",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/orion-platform",
    project_urls={
        "Bug Tracker": "https://github.com/your-org/orion-platform/issues",
        "Documentation": "https://orion-platform.readthedocs.io",
        "Source Code": "https://github.com/your-org/orion-platform",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.12.1",
            "flake8>=6.1.0",
            "mypy>=1.7.1",
            "isort>=5.13.2",
            "pre-commit>=3.6.0",
        ],
        "docs": [
            "sphinx>=7.2.6",
            "sphinx-rtd-theme>=2.0.0",
            "sphinx-autodoc-typehints>=1.25.2",
            "myst-parser>=2.0.0",
        ],
        "gpu": [
            "torch>=2.1.0+cu118",
            "faiss-gpu>=1.7.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "orion=orion.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "orion": [
            "templates/**/*.j2",
            "config/*.yaml",
            "data/*.json",
        ],
    },
    zip_safe=False,
)