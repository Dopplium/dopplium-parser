from setuptools import setup, find_packages

setup(
    name="dopplium-parser",
    version="1.4.0",
    description="Parsers for Dopplium radar data formats (ADCData, RDCh, RadarCube, Detections, Blobs, Tracks)",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20",
        "matplotlib>=3.5",
    ],
    python_requires=">=3.7",
    author="Dopplium",
)

