"""CV karşılaştırma ve değerlendirme sistemi web arayüzü."""

import streamlit as st
import os
import pandas as pd
from cv_parser import parse_cv
from data_extractor import extract_structured_data
from comparison_engine import compare_cv_data, generate_report
from typing import Dict, Any, List

if not os.path.exists("data"):
    os.makedirs("data")

st.set_page_config(layout="wide", page_title="Akıllı CV Karşılaştırma Sistemi")


def run_full_analysis(cv_file, name: str) -> Dict[str, Any]:
    temp_path = os.path.join("data", f"{name}_{cv_file.name}")
    with open(temp_path, "wb") as f:
        f.write(cv_file.getbuffer())

    sections = parse_cv(temp_path)
    if not sections:
        return None
    return extract_structured_data(sections)


st.title("👨‍💻 CV Karşılaştırma ve Değerlendirme Sistemi")
st.subheader("Birden fazla CV yükleyip karşılaştırabilirsiniz.")

num_cvs = st.slider("Kaç CV yüklenecek? (En az 2, en fazla 20)", min_value=2, max_value=20, value=2)

uploaded_files = []
cols = st.columns(2)
for i in range(num_cvs):
    col = cols[i % 2]
    with col:
        uploaded = st.file_uploader(f"CV {i+1} Dosyasını Yükleyin (PDF)", type=["pdf"], key=f"cv_uploader_{i}")
        uploaded_files.append(uploaded)

uploaded_present = [f for f in uploaded_files if f is not None]
if len(uploaded_present) < 2:
    st.info("Lütfen en az 2 adet CV yükleyin.")
else:
    if st.button("🚀 Karşılaştırmayı Başlat", type="primary"):
        with st.spinner("CV'ler parse ediliyor ve analiz ediliyor..."):
            data_list = []
            labels = []
            for idx, f in enumerate(uploaded_present):
                label = chr(65 + idx)
                d = run_full_analysis(f, label)
                data_list.append(d)
                labels.append(label)

        paired = []
        for i in range(len(data_list)):
            if not data_list[i]:
                continue
            filename = uploaded_present[i].name
            display = os.path.splitext(filename)[0]
            paired.append((labels[i], filename, display, data_list[i]))
        if len(paired) < 2:
            st.error("Yüklenen dosyalardan en az iki tanesi okunabilir olmalı.")
        else:
            def count_for_section(data, section_key):
                v = data.get(section_key)
                if v is None:
                    return 0
                if isinstance(v, str):
                    return len(v)
                if isinstance(v, list):
                    return len(v)
                try:
                    return len(v)
                except Exception:
                    return 1

            comparisons = []
            n = len(paired)
            for i in range(n):
                for j in range(i + 1, n):
                    label_i, filename_i, display_i, data_i = paired[i]
                    label_j, filename_j, display_j, data_j = paired[j]
                    pair_label = f"{display_i} vs {display_j}"
                    total_score, section_scores = compare_cv_data(data_i, data_j)
                    report_lines = generate_report(data_i, data_j, total_score, section_scores)
                    comparisons.append((pair_label, total_score, section_scores, report_lines, display_i, display_j, data_i, data_j))

            agg_scores = {}
            all_sections = set()
            for comp in comparisons:
                section_scores = comp[2]
                for s in section_scores.keys():
                    all_sections.add(s)
            ordered_keys = ['DENEYİM', 'YETENEKLER', 'TEKNİK_BECERİLER', 'EĞİTİM', 'YABANCI_DİL', 'SERTİFİKALAR', 'KURSLAR', 'ÖZET']
            for s in all_sections:
                if s not in ordered_keys:
                    ordered_keys.append(s)

            if 'ÖZET' in ordered_keys:
                ordered_keys = [k for k in ordered_keys if k != 'ÖZET']
                insert_index = min(7, len(ordered_keys))
                ordered_keys.insert(insert_index, 'ÖZET')

            for section in ordered_keys:
                vals = [comp[2].get(section, 0.0) for comp in comparisons]
                agg_scores[section] = sum(vals) / len(vals) if vals else 0.0

            rows = []
            for section in ordered_keys:
                row = {'Alan': section, 'Benzerlik Skoru': f"% {agg_scores.get(section,0.0)*100:.1f}"}
                for i, (lbl, filename, display, data) in enumerate(paired):
                    col_name = f"{display} Öğeleri"
                    row[col_name] = count_for_section(data, section)
                rows.append(row)

            idx_ozet = next((i for i, r in enumerate(rows) if r.get('Alan') == 'ÖZET'), None)
            idx_kisi = next((i for i, r in enumerate(rows) if r.get('Alan') == 'KİŞİSEL_BECERİLER'), None)
            if idx_ozet is not None and idx_kisi is not None:
                rows[idx_ozet], rows[idx_kisi] = rows[idx_kisi], rows[idx_ozet]

            scores_df = pd.DataFrame(rows)

            candidate_cols = [f"{display} Öğeleri" for _, filename, display, _ in paired]
            cols_order = ['Alan', 'Benzerlik Skoru'] + candidate_cols
            for c in scores_df.columns:
                if c not in cols_order:
                    cols_order.append(c)
            scores_df = scores_df[cols_order]

            st.table(scores_df)

            st.header("✅ Analiz Tamamlandı")
            total_vals = [comp[1] for comp in comparisons]
            combined_label = " vs ".join([display for _, filename, display, _ in paired])
            avg_total = sum(total_vals) / len(total_vals) if total_vals else 0.0
            st.metric(label=f"Genel Benzerlik ({combined_label})", value=f"% {avg_total*100:.1f}")
            st.markdown("---")

            st.subheader("İK Uzmanı Raporları")
            
            def same_and_diff(list_a, list_b):
                set_a = set([str(x).strip().lower() for x in list_a if x])
                set_b = set([str(x).strip().lower() for x in list_b if x])
                common = sorted(list(set_a & set_b))
                only_a = sorted(list(set_a - set_b))
                only_b = sorted(list(set_b - set_a))
                return common, only_a, only_b

            for pair_item in comparisons:
                pair_label, _, _, report_lines, display_i, display_j, data_i, data_j = pair_item
                with st.expander(pair_label, expanded=False):
                    st.write("**İK Uzmanı Raporu (detay)**")
                    for line in report_lines:
                        st.write(line)

                    st.markdown("---")
                    st.write("**Aynı / Farklı Özellikler**")
                    list_keys = [
                        ("YETENEKLER", lambda d: d.get("YETENEKLER", [])),
                        ("TEKNİK_BECERİLER", lambda d: d.get("TEKNİK_BECERİLER", [])),
                        ("PROJELER", lambda d: [p.get('Raw_Entry') if isinstance(p, dict) else p for p in d.get('PROJELER', [])]),
                        ("SERTİFİKALAR", lambda d: [p.get('Raw_Entry') if isinstance(p, dict) else p for p in d.get('SERTİFİKALAR', [])]),
                        ("KURSLAR", lambda d: [p.get('Raw_Entry') if isinstance(p, dict) else p for p in d.get('KURSLAR', [])]),
                        ("KİŞİSEL_BECERİLER", lambda d: d.get('KİŞİSEL_BECERİLER', [])),
                        ("YABANCI_DİL", lambda d: [ (x.get('dil') if isinstance(x, dict) else x) for x in d.get('YABANCI_DİL', []) ])
                    ]

                    for key, extractor in list_keys:
                        a_list = extractor(data_i) or []
                        b_list = extractor(data_j) or []
                        common, only_a, only_b = same_and_diff(a_list, b_list)
                        st.markdown(f"**{key}**")
                        st.write(f"Ortak ({len(common)}): {', '.join(common) if common else 'Yok'}")
                        st.write(f"{display_i} ({len(only_a)}): {', '.join(only_a) if only_a else 'Yok'}")
                        st.write(f"{display_j} ({len(only_b)}): {', '.join(only_b) if only_b else 'Yok'}")
                        st.markdown("")
            
            all_certs = []
            for _, _, display, data in paired:
                certs = data.get("SERTİFİKALAR", []) + data.get("KURSLAR", [])
                for c in certs:
                    entry = c.get('Raw_Entry') if isinstance(c, dict) else str(c)
                    all_certs.append((display, entry))

            with st.expander("Tüm Kurslar / Sertifikalar", expanded=False):
                if all_certs:
                    for display, entry in all_certs:
                        st.write(f"{display}: {entry}")
                else:
                    st.write("Yok")

            all_refs = []
            for _, _, display, data in paired:
                refs = data.get("REFERANSLAR", [])
                for r in refs:
                    entry = r if not isinstance(r, dict) else (r.get('name') or r.get('raw') or str(r))
                    all_refs.append((display, entry))

            with st.expander("Tüm Referanslar", expanded=False):
                if all_refs:
                    for display, entry in all_refs:
                        st.write(f"{display}: {entry}")
                else:
                    st.write("Yok")