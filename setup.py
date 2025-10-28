"""
Setup script pour la bibliothèque de programmation compétitive
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="competitive-programming",
    version="1.0.0",
    author="Competitive Programming Library",
    author_email="",
    description="Bibliothèque complète d'algorithmes pour programmation compétitive",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education",
    ],
    python_requires=">=3.9",
    install_requires=[
        # Pas de dépendances externes
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "flake8>=4.0.0",
            "black>=22.0.0",
        ],
    },
)

