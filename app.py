# -*- coding: utf-8 -*-
if emails_found:
for em in sorted(emails_found):
all_rows.append({
"E-mail": em,
"Título do Artigo": article_title or "N/A",
"URL do Artigo": article_url,
"URL do PDF": pdf_url or "(não encontrado)",
})


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
df_unique = (
df.sort_values("E-mail")
.drop_duplicates(subset=["E-mail"]) # mantém a primeira ocorrência
.loc[:, ["E-mail"]]
)
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
