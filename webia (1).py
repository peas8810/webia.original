# -*- coding: utf-8 -*-
"""
webia.py — CitacaoMiner adaptado para o 'webia'
Rodar com:  streamlit run webia.py
"""

import os
import re
import textwrap
from typing import List, Tuple, Optional

import requests
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from unidecode import unidecode

# =================== CONFIG ===================
DEFAULT_TOP_N = 30
TIMEOUT = 30
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Safari/537.36"}

# =================== UTIL ===================
def extract_keywords(text: str, k: int = 12) -> List[str]:
    text = unidecode((text or "").lower())
    text = re.sub(r"\s+", " ", text)
    if not text.strip():
        text = "scientific citations openalex crossref pubmed bibliometrics metadata ojs google scholar merge"
    corpus = [text]
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words="english")
    X = vectorizer.fit_transform(corpus)
    feats = vectorizer.get_feature_names_out()
    scores = X.toarray()[0]
    ranked = sorted(list(zip(feats, scores)), key=lambda x: x[1], reverse=True)
    return [w for w,_ in ranked[:k]]

# =================== OPENALEX ===================
def reconstruct_openalex_abstract(inv):
    if not inv:
        return None
    maxpos = 0
    for positions in inv.values():
        if positions:
            maxpos = max(maxpos, max(positions))
    arr = [""] * (maxpos + 1)
    for word, positions in inv.items():
        for pos in positions:
            if 0 <= pos < len(arr):
                arr[pos] = word
    return " ".join([w for w in arr if w])

def openalex_search(keywords: List[str], per_page: int = 50, top_n: int = 50) -> pd.DataFrame:
    q = " ".join(keywords[:8])
    url = "https://api.openalex.org/works"
    params = {"search": q, "sort": "cited_by_count:desc", "per_page": min(per_page, 200)}
    works = []
    page = 1
    while len(works) < top_n:
        params["page"] = page
        r = requests.get(url, params=params, timeout=TIMEOUT, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        results = data.get("results", [])
        if not results:
            break
        for w in results:
            works.append({
                "source": "OpenAlex",
                "id": w.get("id"),
                "title": w.get("title"),
                "year": w.get("publication_year"),
                "cited_by_count": w.get("cited_by_count", 0),
                "doi": w.get("doi"),
                "abstract": reconstruct_openalex_abstract(w.get("abstract_inverted_index")),
                "venue": (w.get("primary_location") or {}).get("source",{}).get("display_name"),
                "oa_url": (w.get("open_access") or {}).get("oa_url") or (w.get("primary_location") or {}).get("landing_page_url")
            })
            if len(works) >= top_n:
                break
        page += 1
    return pd.DataFrame(works)

# =================== CROSSREF ===================
def clean_crossref_abstract(a: Optional[str]) -> Optional[str]:
    if not a:
        return None
    a = re.sub(r"<[^>]+>", " ", a)
    a = re.sub(r"\s+", " ", a).strip()
    return a

def crossref_search(keywords: List[str], rows: int = 50) -> pd.DataFrame:
    q = " ".join(keywords[:8])
    url = "https://api.crossref.org/works"
    params = {"query": q, "rows": rows, "sort": "is-referenced-by-count", "order": "desc"}
    r = requests.get(url, params=params, timeout=TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    items = r.json().get("message", {}).get("items", [])
    recs = []
    for it in items:
        title = " ".join(it.get("title", [])).strip()
        recs.append({
            "source": "Crossref",
            "id": it.get("DOI"),
            "title": title if title else None,
            "year": (it.get("issued",{}).get("date-parts") or [[None]])[0][0],
            "cited_by_count": it.get("is-referenced-by-count", 0),
            "doi": it.get("DOI"),
            "abstract": clean_crossref_abstract(it.get("abstract")),
            "venue": (it.get("container-title") or [None])[0],
            "oa_url": it.get("URL")
        })
    return pd.DataFrame(recs)

# =================== PUBMED ===================
def pubmed_search(keywords: List[str], retmax: int = 50) -> pd.DataFrame:
    q = "+".join(keywords[:8])
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    url = f"{base}esearch.fcgi"
    params = {"db":"pubmed", "term": q, "retmax": retmax, "sort":"relevance", "retmode":"json"}
    r = requests.get(url, params=params, timeout=TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return pd.DataFrame()
    url = f"{base}esummary.fcgi"
    params = {"db":"pubmed", "id": ",".join(ids), "retmode":"json"}
    r = requests.get(url, params=params, timeout=TIMEOUT, headers=HEADERS)
    r.raise_for_status()
    data = r.json().get("result", {})
    recs = []
    for pid in ids:
        it = data.get(pid, {})
        recs.append({
            "source": "PubMed",
            "id": pid,
            "title": it.get("title"),
            "year": (it.get("pubdate") or "").split(" ")[0],
            "cited_by_count": None,
            "doi": (it.get("elocationid") or "").replace("doi: ", "") if it.get("elocationid") else None,
            "abstract": None,
            "venue": it.get("fulljournalname"),
            "oa_url": f"https://pubmed.ncbi.nlm.nih.gov/{pid}/"
        })
    return pd.DataFrame(recs)

# =================== CONSOLIDATE & SIMILARITY ===================
def consolidate(*dfs) -> pd.DataFrame:
    frames = [d for d in dfs if isinstance(d, pd.DataFrame) and len(d)]
    if not frames:
        return pd.DataFrame(columns=["source","id","title","year","cited_by_count","doi","abstract","venue","oa_url"])
    df = pd.concat(frames, ignore_index=True)
    if "cited_by_count" in df.columns:
        df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce")
    return df

def similarity_rank(reference_text: str, df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    texts = []
    for _, row in df.iterrows():
        t = " ".join([str(row.get("title") or ""), str(row.get("abstract") or "")])
        texts.append(t)
    vectorizer = TfidfVectorizer(max_features=8000, stop_words="english")
    X = vectorizer.fit_transform([reference_text] + texts)
    sims = cosine_similarity(X[0:1], X[1:]).flatten()
    out = df.copy()
    out["similarity"] = sims
    out = out.sort_values(["similarity","cited_by_count"], ascending=[False, False])
    return out.head(k)

def suggest_title_and_abstract(reference_text: str, top_df: pd.DataFrame) -> Tuple[str, str]:
    kws = extract_keywords(reference_text, k=8)
    base = " ".join(w.capitalize() for w in kws[:3]) or "Literatura Cientifica"
    method_hint = "Uma Análise de Artigos Altamente Citados em Bases Abertas"
    themes = ", ".join([w for w in kws[:6]])
    top_titles = "; ".join([t for t in top_df["title"].dropna().head(3)]) if not top_df.empty else ""
    abstract = (f"Objetivo: mapear literatura altamente citada relacionada a {themes}. "
                f"Métodos: consulta às APIs OpenAlex, Crossref e PubMed, com ranqueamento por citações e similaridade TF-IDF ao documento de referência. "
                f"Resultados: os títulos mais correlatos incluem {top_titles}. "
                f"Conclusão: a combinação de métricas de citação e similaridade de conteúdo permite identificar rapidamente artigos de referência e lacunas para estudos de revisão e posicionamento editorial.")
    title = f"{base}: {method_hint}"
    return title, textwrap.fill(abstract, 120)

# =================== STREAMLIT UI ===================
def webia_app():
    st.set_page_config(page_title="webia – CitacaoMiner", layout="wide")
    st.markdown("# webia — CitacaoMiner (OpenAlex/Crossref/PubMed)")
    st.write("Minere artigos mais citados (APIs abertas), meça similaridade com seu texto e gere **título+resumo** sugeridos.")
    with st.sidebar:
        st.header("Configuração")
        top_n = st.number_input("Qtde por base (Top N)", 5, 200, DEFAULT_TOP_N, step=5)
        use_crossref = st.checkbox("Incluir Crossref", value=True)
        use_pubmed = st.checkbox("Incluir PubMed", value=True)
        force_query = st.text_input("Tema (opcional – sobrescreve do texto)")
        run_btn = st.button("🚀 Rodar Mineração")

    st.subheader("Documento de referência")
    col1, col2 = st.columns(2)
    with col1:
        user_text = st.text_area("Cole o texto base (ou envie um arquivo ao lado):", height=220)
    with col2:
        uploaded = st.file_uploader("Ou envie um .txt", type=["txt"])
        if uploaded is not None and not user_text.strip():
            try:
                user_text = uploaded.read().decode("utf-8", errors="ignore")
            except Exception:
                user_text = uploaded.read().decode("latin-1", errors="ignore")

    if run_btn:
        try:
            with st.spinner("Extraindo palavras-chave..."):
                if force_query.strip():
                    kws = extract_keywords(force_query, k=12)
                else:
                    kws = extract_keywords(user_text, k=12)
            st.write("**Palavras-chave**:", ", ".join(kws))

            dataframes = []

            with st.spinner("Consultando OpenAlex..."):
                df_oa = openalex_search(kws, top_n=int(top_n))
                dataframes.append(df_oa)
            st.success(f"OpenAlex OK — {len(df_oa)} registros")
            st.dataframe(df_oa.head(10), use_container_width=True)

            if use_crossref:
                with st.spinner("Consultando Crossref..."):
                    df_cr = crossref_search(kws, rows=int(top_n))
                    dataframes.append(df_cr)
                st.success(f"Crossref OK — {len(df_cr)} registros")
                st.dataframe(df_cr.head(10), use_container_width=True)
            else:
                df_cr = pd.DataFrame()

            if use_pubmed:
                with st.spinner("Consultando PubMed..."):
                    df_pm = pubmed_search(kws, retmax=int(top_n))
                    dataframes.append(df_pm)
                st.success(f"PubMed OK — {len(df_pm)} registros")
                st.dataframe(df_pm.head(10), use_container_width=True)
            else:
                df_pm = pd.DataFrame()

            df_all = consolidate(*dataframes)
            if df_all.empty:
                st.warning("Nenhum registro retornado.")
            else:
                st.subheader("Consolidado e ranqueado por citações")
                df_ranked = df_all.sort_values(by=["cited_by_count"], ascending=[False], na_position="last").reset_index(drop=True)
                st.dataframe(df_ranked.head(20), use_container_width=True)

                csv_all = df_ranked.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Baixar CSV (ranking completo)", csv_all, file_name="citacao_top.csv", mime="text/csv")

                st.subheader("Mais similares ao seu texto")
                df_sim = similarity_rank(user_text, df_ranked, k=15)
                st.dataframe(df_sim[["source","title","year","cited_by_count","oa_url","similarity"]], use_container_width=True)
                csv_sim = df_sim.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Baixar CSV (mais similares)", csv_sim, file_name="citacao_semelhantes.csv", mime="text/csv")

                st.subheader("Sugestão de título e resumo")
                sug_title, sug_abs = suggest_title_and_abstract(user_text, df_sim if len(df_sim) else df_ranked)
                st.markdown(f"**Título sugerido:** {sug_title}")
                st.text_area("Resumo sugerido:", value=sug_abs, height=200)

            st.caption("⚖️ Ética & Compliance: Não faz scraping do Google Scholar. Usa OpenAlex/Crossref/PubMed conforme termos.")

        except Exception as e:
            st.error(f"Erro durante a mineração: {e}")
            st.stop()
    else:
        st.info("Cole o texto ou envie um .txt, ajuste as opções e clique em **Rodar Mineração**.")

# Execução direta: streamlit run webia.py
if __name__ == "__main__":
    webia_app()
