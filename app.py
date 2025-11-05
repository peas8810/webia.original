# -*- coding: utf-8 -*-
"""
OJS Email Miner — Streamlit

Adaptação para uma aplicação Streamlit que:
1) Recebe a URL de uma EDIÇÃO (issue) do OJS.
2) Busca os artigos da edição, tenta localizar o PDF (galley).
3) Baixa o PDF e extrai e-mails do conteúdo textual.
4) (Opcional) Usa OCR como fallback se o PDF for imagem/escaneado.

Observações importantes sobre OCR:
- OCR requer dependências de SISTEMA: Tesseract e Poppler.
  • Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y tesseract-ocr libtesseract-dev poppler-utils
  • Windows: instalar Tesseract (UB Mannheim) e Poppler for Windows e adicionar ao PATH
  • macOS (Homebrew): brew install tesseract poppler

Se não houver OCR no ambiente, o app funciona com pdfminer (sem OCR).
"""

import io
import re
import time
from pathlib import Path
from typing import List, Set, Tuple, Optional
from urllib.parse import urljoin, urlparse
from datetime import datetime

import requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text_to_fp

# =====================================================
# Configurações gerais
# =====================================================
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) Chrome/120.0.0.0 Safari/537.36"
}
EMAIL_REGEX = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
CACHE_DIR = Path("cache_pdfs")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
REQUEST_TIMEOUT = 40

# Tenta habilitar OCR (opcional)
try:
    import pytesseract  # type: ignore
    from pdf2image import convert_from_bytes  # type: ignore

    OCR_RUNTIME_AVAILABLE = True
except Exception:
    OCR_RUNTIME_AVAILABLE = False


# =====================================================
# Funções utilitárias de rede / parsing
# =====================================================
def get_soup(url: str) -> BeautifulSoup:
    resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def find_article_links(issue_url: str) -> List[Tuple[str, str]]:
    """
    Retorna lista (article_url, titulo_estimado) a partir da página da edição (issue).
    """
    soup = get_soup(issue_url)
    links: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    # Seletor padrão do OJS para páginas de edição (issue)
    for a in soup.select('a[href*="/article/view/"]'):
        href = a.get("href")
        if not href:
            continue
        full = urljoin(issue_url, href)
        title = a.get_text(strip=True) or None
        key = full.split("?")[0]
        if key not in seen:
            seen.add(key)
            links.append((full, title or "(Sem título)"))

    # Caso a edição use outro template, tente um fallback amplo (com cuidado)
    if not links:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/article/view/" in href:
                full = urljoin(issue_url, href)
                title = a.get_text(strip=True) or "(Sem título)"
                key = full.split("?")[0]
                if key not in seen:
                    seen.add(key)
                    links.append((full, title))

    return links


def find_pdf_link(article_url: str) -> Optional[str]:
    """
    Tenta identificar o link do PDF na página do artigo do OJS.
    """
    soup = get_soup(article_url)

    # OJS 3.x costuma ter essas classes
    for a in soup.select("a.obj_galley_link, a.galley-link"):
        label = a.get_text(" ", strip=True).lower()
        if "pdf" in label:
            href = a.get("href")
            if href:
                return urljoin(article_url, href)

    # Fallback heurístico — alguns OJS usam caminhos numéricos para o galley PDF
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parts = urlparse(href).path.rstrip("/").split("/")
        if len(parts) >= 6 and parts[-2].isdigit() and parts[-1].isdigit():
            return urljoin(article_url, href)

    # Outro fallback: procurar por links que terminem com .pdf
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.lower().endswith(".pdf"):
            return urljoin(article_url, href)

    return None


def cache_path_for_pdf(pdf_url: str) -> Path:
    safe = re.sub(
        r"[^A-Za-z0-9._-]+",
        "_",
        urlparse(pdf_url).path + "_" + (urlparse(pdf_url).query or ""),
    )
    return CACHE_DIR / f"{safe or 'file'}.pdf"


def download_pdf(pdf_url: str) -> bytes:
    """
    Baixa o PDF com cache local simples. Tenta com e sem parâmetro 'download=1'.
    """
    cpath = cache_path_for_pdf(pdf_url)
    if cpath.exists() and cpath.stat().st_size > 0:
        return cpath.read_bytes()

    try_urls = [pdf_url]
    if "download=" not in pdf_url:
        sep = "&" if "?" in pdf_url else "?"
        try_urls.append(f"{pdf_url}{sep}download=1")

    for u in try_urls:
        resp = requests.get(u, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200 and resp.content:
            cpath.write_bytes(resp.content)
            return resp.content
    raise RuntimeError(f"Falha ao baixar PDF {pdf_url}")


# =====================================================
# Extração de texto / e-mails
# =====================================================
def extract_text_pdfminer(pdf_bytes: bytes) -> str:
    output = io.StringIO()
    try:
        with io.BytesIO(pdf_bytes) as bio:
            extract_text_to_fp(bio, output, laparams=None, output_type="text", codec=None)
        return output.getvalue()
    except Exception:
        return ""


def extract_text_ocr(pdf_bytes: bytes, dpi: int) -> str:
    if not OCR_RUNTIME_AVAILABLE:
        return ""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
        texts = []
        for img in images:
            # Português + Inglês para aumentar recall
            txt = pytesseract.image_to_string(img, lang="eng+por")
            texts.append(txt)
        return "\n".join(texts)
    except Exception:
        return ""


def extract_emails_from_text(text: str) -> Set[str]:
    emails = set(EMAIL_REGEX.findall(text))
    # higieniza pontuações comuns
    return {e.strip(".,;:·/\\()[]{}<>") for e in emails}


def extract_emails_from_pdf(pdf_bytes: bytes, ocr_fallback: bool, ocr_dpi: int) -> Set[str]:
    # 1) Tenta via texto
    text = extract_text_pdfminer(pdf_bytes)
    emails = extract_emails_from_text(text)
    if emails:
        return emails

    # 2) (Opcional) Tenta via OCR
    if ocr_fallback and OCR_RUNTIME_AVAILABLE:
        ocr_text = extract_text_ocr(pdf_bytes, ocr_dpi)
        emails = extract_emails_from_text(ocr_text)
        return emails

    return set()


# =====================================================
# UI — Streamlit
# =====================================================
st.set_page_config(page_title="OJS Email Miner", page_icon="📧", layout="wide")

st.title("📧 OJS Email Miner — Streamlit")
st.caption("Insira a URL de uma edição (issue) do OJS para minerar e-mails dos PDFs dos artigos.")

with st.sidebar:
    st.header("Configurações")
    issue_url = st.text_input(
        "URL da edição (issue)",
        placeholder="https://ojs.seudominio/ojs/index.php/suarevista/issue/view/123",
    )
    pause_seconds = st.slider("Intervalo entre artigos (seg)", min_value=0.0, max_value=5.0, value=1.0, step=0.1)

    ocr_requested = st.checkbox(
        "Usar OCR (para PDFs escaneados)",
        value=False,
        help="Necessita Tesseract e Poppler instalados no SISTEMA. Se não estiverem disponíveis, o app continua sem OCR.",
    )
    ocr_dpi = st.slider("DPI do OCR", min_value=150, max_value=400, value=250, step=50)

    st.divider()
    st.markdown(
        f"**OCR disponível no ambiente:** {'✅ Sim' if OCR_RUNTIME_AVAILABLE else '❌ Não'}  \n"
        "Caso esteja ❌, instale Tesseract + Poppler no sistema para ativar."
    )

col_run, col_about = st.columns([1, 1])

with col_about:
    with st.expander("Como usar / Notas importantes", expanded=False):
        st.markdown(
            """
            1. Cole a **URL completa** da *edição (issue)* do OJS.
            2. Ajuste o intervalo entre requisições para evitar sobrecarga do site.
            3. **Opcional**: Ative **OCR** se os PDFs forem imagens (escaneados). Requer Tesseract + Poppler no sistema.
            4. Clique em **Iniciar Mineração** e aguarde o progresso.

            **Exportação:** ao final, baixe o CSV com os e-mails únicos ou a planilha completa.
            """
        )

run = col_run.button("🚀 Iniciar Mineração", use_container_width=True)
status = st.empty()
progress = st.progress(0)
results_container = st.container()

if run:
    if not issue_url or not issue_url.startswith(("http://", "https://")):
        st.error("Insira uma URL válida que comece com http:// ou https://")
        st.stop()

    try:
        status.info("Buscando artigos na edição…")
        articles = find_article_links(issue_url)
        if not articles:
            st.warning("Nenhum artigo foi encontrado nessa edição.")
            st.stop()

        status.success(f"Encontrados {len(articles)} artigos. Iniciando processamento…")

        all_rows = []
        processed = 0
        progress_total = max(1, len(articles))

        for article_url, article_title in articles:
            try:
                pdf_url = find_pdf_link(article_url)
                emails_found: Set[str] = set()

                if pdf_url:
                    pdf_bytes = download_pdf(pdf_url)
                    emails_found = extract_emails_from_pdf(
                        pdf_bytes=pdf_bytes,
                        ocr_fallback=bool(ocr_requested),
                        ocr_dpi=int(ocr_dpi),
                    )

                if emails_found:
                    for em in sorted(emails_found):
                        all_rows.append(
                            {
                                "E-mail": em,
                                "Título do Artigo": article_title or "N/A",
                                "URL do Artigo": article_url,
                                "URL do PDF": pdf_url or "(não encontrado)",
                            }
                        )

            except Exception as e:
                st.warning(f"Erro no artigo: {article_url}\n{e}")

            processed += 1
            progress.progress(int(processed * 100 / progress_total))
            time.sleep(float(pause_seconds))

        # --- Resultados ---
        with results_container:
            st.divider()
            st.subheader("📊 Resultados da Mineração")

            if all_rows:
                df = pd.DataFrame(all_rows)
                unique_count = df["E-mail"].nunique()

                c1, c2, c3 = st.columns(3)
                c1.metric("E-mails (linhas)", len(df))
                c2.metric("E-mails únicos", unique_count)
                c3.metric("Artigos processados", f"{processed}/{len(articles)}")

                st.markdown("### Tabela completa")
                st.dataframe(df, use_container_width=True, hide_index=True)

                # Downloads
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                csv_full = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Baixar CSV (completo)",
                    data=csv_full,
                    file_name=f"emails_ojs_{ts}.csv",
                    mime="text/csv",
                )

                # CSV de únicos
                df_unique = df.sort_values("E-mail").drop_duplicates(subset=["E-mail"]).loc[:, ["E-mail"]]
                csv_unique = df_unique.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Baixar CSV (e-mails únicos)",
                    data=csv_unique,
                    file_name=f"emails_unicos_ojs_{ts}.csv",
                    mime="text/csv",
                )

                with st.expander("👀 Preview (até 100 primeiros únicos)"):
                    st.write("\n".join(df_unique["E-mail"].head(100).tolist()))
            else:
                st.info("Nenhum e-mail foi encontrado nos PDFs processados.")
                st.markdown(
                    "- Ative **OCR** se os PDFs forem escaneados.\n"
                    "- Verifique se os PDFs realmente contêm e-mails em texto.\n"
                    "- Tente outra edição."
                )

    except Exception as e:
        st.error("Erro durante a mineração. Verifique a URL, a disponibilidade pública da edição e a conexão.")
        st.exception(e)
