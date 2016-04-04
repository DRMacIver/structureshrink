from setuptools import find_packages, setup
import os


def local_file(name):
    return os.path.relpath(os.path.join(os.path.dirname(__file__), name))

SOURCE = local_file("src")
README = local_file("README.rst")


setup(
    name='structureshrink',
    version="0.0.1",
    author='David R. MacIver',
    author_email='david@drmaciver.com',
    packages=find_packages(SOURCE),
    package_dir={"": SOURCE},
    url='https://github.com/DRMacIver/hypothesis',
    license='AGPL v3',
    description='A library for property based testing',
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    install_requires=['click'],
    long_description=open(README).read(),
    entry_points={
        'console_scripts': [
            'structureshrink=structureshrink.__main__:shrinker'
        ]
    }
)
