"""
Setup script for Security Framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="security-framework",
    version="1.0.0",
    description="A comprehensive Python framework for firewall testing and DNS reconnaissance",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Security Framework Team",
    packages=find_packages(),
    install_requires=[
        "python-nmap>=1.6.0",
        "dnspython>=2.4.0",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "security-framework=security_framework.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Information Technology",
        "Topic :: Security",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
    ],
    keywords="security testing firewall dns reconnaissance penetration testing",
)
