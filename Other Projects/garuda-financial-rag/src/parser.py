import os
from dotenv import load_dotenv
import pdfplumber
from langchain_core.documents import Document

load_dotenv()

DEFAULT_START_PAGE = 525
DEFAULT_END_PAGE = 689


def _page_range(start_page: int, end_page: int, total_pages: int):
    if start_page <= 0:
        raise ValueError("START_PAGE harus lebih besar dari 0.")
    if end_page < start_page:
        raise ValueError("END_PAGE harus sama atau lebih besar dari START_PAGE.")

    zero_based_start = max(0, start_page - 1)
    zero_based_end = max(0, end_page - 1)

    if zero_based_start >= total_pages:
        raise ValueError(f"START_PAGE {start_page} melebihi total halaman PDF ({total_pages}).")

    zero_based_end = min(zero_based_end, total_pages - 1)
    return zero_based_start, zero_based_end


def _extract_page_text(page):
    # struktur kolom pada laporan keuangan.
    text = page.extract_text(layout=True, x_tolerance=2, y_tolerance=2) or ""
    if not text.strip():
        # Fallback: rekontruksi dari words yg biasa gagal.
        words = page.extract_words(
            use_text_flow=True,
            keep_blank_chars=False,
            x_tolerance=2,
            y_tolerance=2
        )
        text = " ".join(word.get("text", "").strip() for word in words if word.get("text"))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def _extract_page_tables(page):
    table_settings = {
        "vertical_strategy": "lines_strict",
        "horizontal_strategy": "lines_strict",
        "intersection_x_tolerance": 5,
        "intersection_y_tolerance": 5,
        "snap_tolerance": 3,
        "join_tolerance": 3,
    }
    tables = page.extract_tables(table_settings=table_settings)
    if not tables:
        # Fallback tidak punya garis
        fallback_settings = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "text_x_tolerance": 2,
            "text_y_tolerance": 2,
        }
        tables = page.extract_tables(table_settings=fallback_settings)
    if not tables:
        return ""

    fragments = []
    for table_index, table in enumerate(tables, start=1):
        rows = []
        for row in table:
            cells = [cell.replace("\n", " ").strip() if cell else "" for cell in row]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            fragments.append(f"TABEL {table_index}:\n" + "\n".join(rows))

    return "\n\n".join(fragments)


def load_targeted_pages():
    pdf_path = "data/garuda_annual_report_2025.pdf"
    start_page = int(os.getenv("START_PAGE", DEFAULT_START_PAGE))
    end_page = int(os.getenv("END_PAGE", DEFAULT_END_PAGE))

    print(f"[*] Menjalankan parser PDF untuk rentang halaman {start_page}-{end_page} dari: {pdf_path}")

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File {pdf_path} tidak ditemukan di folder data/!")

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        zero_start, zero_end = _page_range(start_page, end_page, total_pages)

        documents = []
        for page_idx in range(zero_start, zero_end + 1):
            page = pdf.pages[page_idx]
            page_number = page_idx + 1
            page_text = _extract_page_text(page)
            table_text = _extract_page_tables(page)

            if not page_text and not table_text:
                page_text = "(halaman kosong atau teks tidak tersedia untuk halaman ini)"

            raw_content = page_text
            if table_text:
                raw_content += "\n\n" + table_text

            clean_lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
            clean_text = "\n".join(clean_lines)

            enhanced_content = (
                f"DOKUMEN: LAPORAN KEUANGAN GARUDA INDONESIA TAHUN BUKU 2025/2024\n"
                f"SOURCE PAGE: {page_number}\n"
                f"KONTEX: Laporan keuangan, tabel, angka, aset, liabilitas, ekuitas, pendapatan.\n"
                f"================================================================================\n"
                f"{clean_text}"
            )

            documents.append(Document(page_content=enhanced_content, metadata={"source": pdf_path, "page": page_number}))

    if not documents:
        raise RuntimeError("Tidak ada halaman yang berhasil diekstrak dari PDF.")

    print(f"[+] Berhasil mengekstrak {len(documents)} halaman dari report.")
    return documents


if __name__ == "__main__":
    pages = load_targeted_pages()
    print("\n--- Cek Hasil Halaman Pertama ---")
    if pages:
        print(pages[0].page_content[:800])