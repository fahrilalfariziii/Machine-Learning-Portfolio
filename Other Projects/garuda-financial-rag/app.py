import streamlit as st
import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.llm_chain import ask_financial_ai

st.set_page_config(page_title="Garuda Financial AI Assistant", page_icon="✈️")
st.title("✈️ Garuda Financial AI Assistant")
st.write("Analisis Laporan Keuangan Garuda Indonesia 2025 - Halaman 525–689.")
st.write("---")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Status Sistem")
    if os.path.isdir("chroma_db"):
        st.success("Database vektor ditemukan.")
    else:
        st.warning("Database vektor belum dibuat. Jalankan 'python src/vectorstore.py' terlebih dahulu.")
    st.markdown("- Model LLM: Groq Llama 3.1-8b\n- Indeks halaman: 525–689\n- PDF: data/garuda_annual_report_2025.pdf")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Masukkan pertanyaan analisis keuangan Anda..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("assistant"):
        with st.spinner("Membaca dokumen dan memverifikasi laporan angka..."):
            try:
                output_text = ask_financial_ai(user_query)
            except Exception as e:
                output_text = (
                    f"[-] Terjadi kesalahan pada sistem: {e}\n\n"
                    "Pastikan 'python src/vectorstore.py' telah dijalankan dan variabel lingkungan 'GROQ_API_KEY' valid."
                )
            st.markdown(output_text)

    st.session_state.messages.append({"role": "assistant", "content": output_text})