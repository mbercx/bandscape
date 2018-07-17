from setuptools import setup, find_packages

setup(
    name="bandscape",
    version="0.1",
    packages=find_packages(exclude=["docs"]),
    install_requires=[
        "pymatgen",
        "numpy",
        "scipy",
        "click",
    ],
    entry_points='''
    '''
)