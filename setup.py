from setuptools import setup


setup(
    name="chazutsu",
    version="0.8.2",
    description="The tool to make NLP datasets ready to use",
    keywords=["machine learning", "nlp", "natural language processing"],
    author="icoxfog417",
    author_email="icoxfog417@yahoo.co.jp",
    license="Apache License 2.0",
    packages=[
        "chazutsu",
        "chazutsu.datasets",
        "chazutsu.datasets.framework"
        ],
    url="https://github.com/chakki-works/chazutsu",
    install_requires=[
        "requests>=2.14.2",
        "tqdm>=4.11.2",
        "pandas>=0.20.1",
        "joblib>=0.11",
    ],
)
