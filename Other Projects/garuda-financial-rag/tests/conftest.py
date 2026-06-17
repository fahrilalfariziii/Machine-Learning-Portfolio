import os

import pytest

PDF_PATH = "data/garuda_annual_report_2025.pdf"
CHROMA_PATH = "chroma_db"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: tests requiring PDF, chroma_db, or GROQ_API_KEY",
    )


@pytest.fixture
def pdf_available():
    return os.path.exists(PDF_PATH)


@pytest.fixture
def chroma_available():
    return os.path.isdir(CHROMA_PATH)


@pytest.fixture
def groq_available():
    return bool(os.getenv("GROQ_API_KEY"))
