from setuptools import setup, find_packages

setup(
    name="model-compatibility-checker",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.29.0",
        "huggingface-hub>=0.19.4",
        "anthropic>=0.7.7",
        "psutil>=5.9.7",
        "gputil>=1.4.0",
        "pandas>=2.1.4",
        "pydantic>=2.5.3",
        "plotly>=5.18.0",
        "python-dotenv>=1.0.0"
    ],
) 