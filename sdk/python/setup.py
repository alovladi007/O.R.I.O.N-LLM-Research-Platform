"""
Setup script for nano-os Python SDK.

Session 28: Python SDK and Workflow DSL
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="nano-os",
    version="0.1.0",
    author="NANO-OS Team",
    author_email="team@nano-os.org",
    description="Python SDK for NANO-OS (Nanomaterials Operating System)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alovladi007/O.R.I.O.N-LLM-Research-Platform",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=[
        "requests>=2.31.0",
        "pyyaml>=6.0",
        "pydantic>=2.0.0",
        "httpx>=0.24.0",
        "tenacity>=8.2.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "black>=23.0.0",
            "mypy>=1.5.0",
            "jupyter>=1.0.0",
            "matplotlib>=3.7.0",
            "numpy>=1.24.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "nano-os=nano_os.cli:main",
        ],
    },
)
