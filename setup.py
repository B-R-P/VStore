from setuptools import setup
import os

# read the long description from README.md
here = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="vstore",
    version="0.1.0",
    author="Bijin Regi Panicker",
    author_email="bijinregipanicker@gmail.com",
    description="Embedded key-value store with vector similarity search",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/B-R-P/VStore",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    py_modules=["vstore"],
    python_requires=">=3.8",
    install_requires=[
        "lmdb",
        "msgpack",
        "fixed-install-nmslib",
        "numpy",
        "scipy",
    ],
)