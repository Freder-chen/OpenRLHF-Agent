from pathlib import Path
from setuptools import find_namespace_packages, setup


BASE_DIR = Path(__file__).parent
README = (
    (BASE_DIR / "README.md").read_text(encoding="utf-8")
    if (BASE_DIR / "README.md").exists()
    else ""
)


setup(
    name="openrlhf-agent",
    version="0.0.1",
    description="Minimal agent runtime primitives for OpenRLHF workflows.",
    long_description=README,
    long_description_content_type="text/markdown",
    author="freder",
    license="Apache-2.0",
    python_requires=">=3.10",
    package_dir={"": "src"},
    packages=find_namespace_packages(where="src"),
    package_data={
        "openrlhf_agent": ["model/protocols/qwen/templates/*.jinja"],
    },
    install_requires=[
        "httpx>=0.27",
        "jinja2>=3.1",
        "openai>=2.0",
        "pydantic>=2.0",
        "pylatexenc>=2.10",
        "sympy>=1.12",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0",
            "pytest-asyncio>=0.23",
            "ruff>=0.5",
            "mypy>=1.10",
        ],
    },
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
