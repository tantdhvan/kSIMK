from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
LONG_DESCRIPTION = (this_directory / "README.md").read_text()

VERSION = '0.0.13'
DESCRIPTION = 'InfluenceDiffusion package'

setup(
    name="InfluenceDiffusion",
    version=VERSION,
    author="Alexander Kagan",
    author_email="<amkagan@umich.edu>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=["numpy",
                      "scipy",
                      "networkx",
                      "typing",
                      "joblib"],

    keywords=['python', 'Influence Maximization', "Network diffusion models",
              "General Linear Threshold model", "Social Networks", "Independent Cascade model"],

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
