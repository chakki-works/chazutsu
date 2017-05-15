from setuptools import setup


setup(
    name = "chazutsu",
    version = "0.3",
    description = "The tool to make NLP datasets ready to use",
    keywords = ["machine learning", "nlp", "natural language processing"],
    author = "icoxfog417",
    author_email = "icoxfog417@yahoo.co.jp",
    license="Apache License 2.0",
    packages = [
        "chazutsu",
        "chazutsu.datasets",
        "chazutsu.datasets.framework"
        ],
    url = "https://github.com/chakki-works/chazutsu",
    download_url = "https://github.com/chakki-works/chazutsu/archive/0.3.tar.gz",
    install_requires=[
        "rarfile>=3.0",
        "requests>=2.14.2",
        "tqdm>=4.11.2",
        "pandas>=0.20.1"
    ],
)