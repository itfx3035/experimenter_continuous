from setuptools import setup, find_packages

setup(
    name="experimenter",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "langchain-ollama>=0.2.0",
        "langchain-core>=0.3.0",
    ],
)