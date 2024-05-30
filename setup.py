import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="granudrum_analysis",
    version="0.0.1",
    author="Ben Jenkins",
    author_email="bdj746@student.bham.ac.uk",
    description="Python package for analysing data from the GranuDrum simulation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/BenDJenkins/granudrum_analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL3.0 License",
        "Operating System :: OS Independent",
    ],
	install_requires=[
        'numpy',
        'opencv-python',
        'imutils',
        'warnings',
        'plotly',
        'seaborn',
        'PIL',
        'collections',
        'pandas'
    ]
)