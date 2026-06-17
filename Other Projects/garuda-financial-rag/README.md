# Garuda Financial RAG

Asisten analisis laporan keuangan berbasis **Retrieval-Augmented Generation (RAG)** untuk Laporan Tahunan Garuda Indonesia. Sistem membaca PDF laporan keuangan, mengekstrak teks dan tabel, lalu menjawab pertanyaan pengguna dengan konteks dokumen yang relevan.

## Fitur

- Ekstraksi PDF laporan keuangan (teks + tabel) dengan `pdfplumber`
- Indeks vektor lokal menggunakan **ChromaDB** dan embedding **sentence-transformers/all-MiniLM-L6-v2**
- Retrieval hibrida: semantic search (MMR) + lexical keyword fallback
- Query expansion untuk istilah keuangan Indonesia/Inggris (aset, liabilitas, pendapatan, arus kas, dll.)
- Jawaban via **Groq Llama 3.1 8B Instant** dengan prompt audit keuangan
- Verifikasi grounding untuk menekan halusinasi 
- Antarmuka chat berbasis **Streamlit**


## Struktur Proyek

```
garuda-financial-rag/
├── app.py                  # Aplikasi Streamlit
├── requirements.txt        # Dependensi Python
├── requirements-dev.txt    # PyTest + Ragas
├── pytest.ini              # Konfigurasi PyTest
├── data/
│   └── garuda_annual_report_2025.pdf
├── chroma_db/              # Vector store (dibuat setelah indexing)
├── eval/
│   ├── golden_dataset.json # Dataset evaluasi RAG
│   ├── run_ragas.py        # Skrip evaluasi Ragas
│   └── results/            # Output laporan evaluasi
├── tests/                  # Unit & integration tests
└── src/
    ├── parser.py           # Parsing PDF per halaman
    ├── vectorstore.py      # Pembuatan & akses vector store
    └── llm_chain.py        # Pipeline RAG + anti-halusinasi
```

## Prasyarat

- Python 3.10+
- API key Groq ([https://console.groq.com](https://console.groq.com))
- File PDF laporan keuangan di `data/garuda_annual_report_2025.pdf`

## Instalasi

1. Clone atau unduh proyek ini, lalu masuk ke direktori proyek.

2. Buat dan aktifkan virtual environment (disarankan):

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

3. Instal dependensi:

```bash
pip install -r requirements.txt
```

4. Buat file `.env` di root proyek:

```env
GROQ_API_KEY=your_groq_api_key_here
START_PAGE=525
END_PAGE=689
```

| Variabel | Deskripsi | Default |
|---|---|---|
| `GROQ_API_KEY` | API key Groq untuk LLM | Wajib |
| `START_PAGE` | Halaman awal PDF yang diindeks (1-based) | `525` |
| `END_PAGE` | Halaman akhir PDF yang diindeks (1-based) | `689` |

## Menyiapkan Data & Vector Store

Pastikan PDF tersedia:

```
data/garuda_annual_report_2025.pdf
```

Bangun ulang indeks vektor:

```bash
python src/vectorstore.py
```

Proses ini akan:
1. Mengekstrak halaman sesuai rentang `START_PAGE`–`END_PAGE`
2. Memecah konten menjadi chunk
3. Menyimpan embedding ke folder `chroma_db/`

Jalankan ulang perintah di atas setiap kali PDF, rentang halaman, atau logika parser berubah.

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

Buka URL lokal yang muncul di terminal (biasanya `http://localhost:8501`).

## Contoh Pertanyaan

- `berapa aset tidak lancar tahun 2025`
- `berapa total liabilitas 2025`
- `berapa pendapatan usaha tahun 2025`
- `berapa operating revenues 2025`
- `bandingkan pendapatan 2025 dan 2024`

Tips agar jawaban lebih akurat:
- Sebutkan **akun** dan **tahun** secara eksplisit
- Gunakan istilah Indonesia atau Inggris (keduanya didukung)
- Untuk perbandingan, sertakan kedua tahun (mis. `2024` dan `2025`)

## Alur RAG (Ringkas)

1. **Parsing** — `parser.py` mengekstrak teks dan tabel dari PDF
2. **Chunking & Indexing** — `vectorstore.py` membuat chunk + metadata halaman
3. **Retrieval** — `llm_chain.py` mengambil chunk relevan (semantic + lexical)
4. **Generation** — LLM menyusun jawaban dari konteks terpilih
5. **Verifikasi** — jawaban dicek agar angka/fakta sesuai konteks PDF

## Pengujian (PyTest)

Instal dependensi pengujian:

```bash
pip install -r requirements-dev.txt
```

Jalankan unit test (cepat, tanpa API):

```bash
pytest -m "not integration"
```

Jalankan semua test termasuk integrasi (butuh PDF + `chroma_db` + `GROQ_API_KEY`):

```bash
pytest
```

| File test | Cakupan |
|---|---|
| `tests/test_parser.py` | Validasi rentang halaman & ekstraksi PDF |
| `tests/test_vectorstore.py` | Chunking dokumen |
| `tests/test_llm_chain.py` | Query expansion, retrieval helper, anti-halusinasi |
| `tests/test_integration.py` | Retrieval end-to-end ke ChromaDB |

## Evaluasi Kualitas RAG (Ragas)

Dataset evaluasi tersedia di `eval/golden_dataset.json` (pertanyaan + ground truth dari PDF Garuda).

Prasyarat:
1. `python src/vectorstore.py` sudah dijalankan
2. `.env` berisi `GROQ_API_KEY`

Jalankan evaluasi:

```bash
python eval/run_ragas.py
```

Opsi:

```bash
# evaluasi 3 pertanyaan pertama saja (lebih cepat)
python eval/run_ragas.py --limit 3
```

Metrik yang diukur:
- **faithfulness** — jawaban sesuai konteks (anti-halusinasi)
- **answer_relevancy** — jawaban relevan dengan pertanyaan
- **context_precision** — konteks yang diambil tepat
- **context_recall** — konteks mencakup informasi yang dibutuhkan

Hasil evaluasi disimpan ke `eval/results/ragas_report_*.json`.

**Catatan kompatibilitas:** Ragas 0.4.x membutuhkan patch import VertexAI yang sudah dihandle otomatis di `eval/run_ragas.py`. Tidak perlu menginstal Google Vertex AI.

## Stack Teknologi

| Komponen | Teknologi |
|---|---|
| UI | Streamlit |
| PDF parsing | pdfplumber |
| Vector DB | ChromaDB |
| Embedding | HuggingFace `all-MiniLM-L6-v2` |
| LLM | Groq `llama-3.1-8b-instant` |
| Orchestration | LangChain |

## Troubleshooting

**`Vector store tidak ditemukan`**
- Jalankan `python src/vectorstore.py` terlebih dahulu.

**`GROQ_API_KEY belum disetel`**
- Pastikan `.env` berisi `GROQ_API_KEY` yang valid.

**Jawaban "tidak tersedia" padahal data ada di PDF**
- Rebuild vector store: `python src/vectorstore.py`
- Perjelas query (nama akun + tahun)
- Cek apakah halaman target berada dalam rentang `START_PAGE`–`END_PAGE`

**Model loading lambat di awal**
- Normal pada run pertama karena model embedding diunduh dari HuggingFace.
