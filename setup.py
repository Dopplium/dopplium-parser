from setuptools import setup, find_packages

setup(
    name="dopplium-parser",
    version="1.1.0",
    description="Parsers for Dopplium radar data formats (RawData and RDCh)",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.5",
    ],
    python_requires=">=3.7",
    author="Dopplium",
)

