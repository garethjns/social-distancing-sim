import setuptools

from social_distancing_sim import __version__


"""
with open("README.md", "r") as fh:
    long_description = fh.read()
"""

setuptools.setup(
    name="social_distancing_sim",
    version=__version__,
    author="Gareth Jones",
    author_email="",
    description="",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/garethjns/social-distancing-sim",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"],
    python_requires='>=3.6',
    install_requires=["networkx", "numpy", "matplotlib", "imageio", "dataclasses", "tqdm", "joblib"]
)