from pathlib import Path

from setuptools import find_packages, setup

long_description = Path("README.md").read_text(encoding="utf-8")

setup(
    name="nDFU",
    version="0.9.1",
    description="Normalized distance from unimodality",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="John Pavlopoulos",
    author_email="annis@aueb.gr",
    url="https://github.com/ipavlopoulos/ndfu",
    packages=find_packages(include=["ndfu", "ndfu.*", "src"]),
    install_requires=["numpy"],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
)
