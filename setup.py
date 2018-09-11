import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="SROMPy",
    version="0.0.1",
    author="James Warner ",
    author_email="james.e.warner@nasa.gov",
    description="Stochastic Reduced Order Models with Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NASA/SROMPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 2",
        "License :: NASA OPEN SOURCE AGREEMENT VERSION 1.3",
        "Operating System :: OS Independent",
    ],
)