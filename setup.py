from glob import glob
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="corpus_graphs",
    version="0.0.1",
    author="Lachlan Dunn",
    author_email="lachlan_dunn6@outlook.com",
    description="Corpus Graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LachlanJDunn/Corpus-Graphs",
    packages=setuptools.find_packages(),
    install_requires=list(open('requirements.txt')),
    classifiers=[],
    python_requires='>=3.6',
    package_data={
        '': ['requirements.txt'],
    },
)
