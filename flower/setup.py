from setuptools import setup, find_packages

setup(
    name="flwr2",
    version="0.1",
    description="Flower with xmk-ckks",
    packages=find_packages(where="src/py"),
    package_dir={"": "src/py"},
)