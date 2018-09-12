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
    packages=["SROMPy",
              "SROMPy.optimize",
              "SROMPy.postprocess",
              "SROMPy.srom",
              "SROMPy.target"],
    package_dir={'SROMPy': 'src'},
    classifiers=[
        "Programming Language :: Python :: 2.7",
        "License :: Freeware",
        "Operating System :: OS Independent",
    ],
)
