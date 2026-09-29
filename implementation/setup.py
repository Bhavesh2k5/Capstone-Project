from setuptools import find_packages, setup

setup(
    name="vljepa-clip-cctv",
    version="1.0.0",
    description="VL-JEPA vs CLIP for CCTV Event Retrieval: efficiency, generalization and faithful report generation",
    packages=find_packages(include=("config", "data", "models", "retrieval", "reporting", "evaluation", "utils")),
    python_requires=">=3.10",
)
