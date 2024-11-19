from setuptools import setup, find_packages

setup(
    name="prompt_steerability",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "matplotlib",
        "openai",
        "pandas",
        "pyyaml",
        "scikit-learn",
        "tqdm",
        "vllm"
    ],
    extras_require={
        'dev': [],
        'docs': []
    }
)
