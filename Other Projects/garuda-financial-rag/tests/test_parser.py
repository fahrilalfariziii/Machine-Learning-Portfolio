import pytest

from src.parser import DEFAULT_END_PAGE, DEFAULT_START_PAGE, _page_range


def test_page_range_valid():
    start, end = _page_range(525, 689, total_pages=800)
    assert start == 524
    assert end == 688


def test_page_range_clamps_to_total_pages():
    start, end = _page_range(525, 900, total_pages=600)
    assert start == 524
    assert end == 599


def test_page_range_invalid_start():
    with pytest.raises(ValueError, match="START_PAGE"):
        _page_range(0, 689, total_pages=800)


def test_page_range_invalid_order():
    with pytest.raises(ValueError, match="END_PAGE"):
        _page_range(700, 600, total_pages=800)


def test_page_range_start_exceeds_pdf():
    with pytest.raises(ValueError, match="melebihi"):
        _page_range(900, 950, total_pages=800)


def test_default_page_constants():
    assert DEFAULT_START_PAGE == 525
    assert DEFAULT_END_PAGE == 689


@pytest.mark.integration
def test_load_targeted_pages_returns_documents(pdf_available):
    if not pdf_available:
        pytest.skip("PDF tidak ditemukan di data/")

    from src.parser import load_targeted_pages

    docs = load_targeted_pages()
    assert len(docs) > 0
    assert all(doc.metadata.get("page") for doc in docs)
    assert all(len(doc.page_content) > 0 for doc in docs)
