""" Configuration file for pypi package."""
from os import path

from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as fp:
    install_requires = fp.read()

setup(
    name="aixi_dataset",
    version="0.1",
    description="XSDS - Synthetic dataset",
    author="Miquel Miró Nicolau, Dr. Gabriel Moyà Alcover, Dr. Antoni Jaume-i-Capó",
    author_email="miquel.miro@uib.cat, gabriel_moya@uib.cat, antoni.jaume@uib.cat",
    license="MIT",
    packages=["aixi"],
    keywords=["XAI", "Deep Learning", "Computer Vision", "Saliency maps"],
    install_requires=install_requires,
    python_requires=">=3.10",
    long_description=long_description,
    long_description_content_type="text/markdown",
    zip_safe=False,
)
