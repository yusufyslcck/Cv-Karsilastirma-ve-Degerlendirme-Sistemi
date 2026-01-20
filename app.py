"""
CV Karşılaştırma ve Değerlendirme Sistemi - Ana Web Arayüzü (Streamlit)
Bu modül, kullanıcı arayüzünü oluşturur, dosya yükleme işlemlerini yönetir
ve analiz sonuçlarını görselleştirir.
"""

import streamlit as st
import os
import pandas as pd
from typing import Dict, Any, List, Union
from rapidfuzz import fuzz  

# --- Yerel Modüller ---
from cv_parser import parse_cv
from data_extractor import extract_structured_data
from comparison_engine import compare_cv_data, generate_report, clean_term

# --- Sayfa ve Klasör Ayarları ---
if not os.path.exists("data"):
    os.makedirs("data")

st.set_page_config(layout="wide", page_title="Akıllı CV Karşılaştırma Sistemi")

# --- Sabitler: Eşleşme Kuralları ---
# Hangi başlığın hangi hassasiyetle (threshold) karşılaştırılacağını belirler.
SECTION_MATCHING_RULES = {
    "KİŞİSEL_BECERİLER": {"type": "fuzzy", "threshold": 70}, 
    "YETENEKLER": {"type": "fuzzy", "threshold": 75},
    "PROJELER": {"type": "fuzzy", "threshold": 65},
    "TEKNİK_BECERİLER": {"type": "fuzzy", "threshold": 85}, 
    "SERTİFİKALAR": {"type": "fuzzy", "threshold": 85},
    "KURSLAR": {"type": "fuzzy", "threshold": 85},
    "YABANCI_DİL": {"type": "exact"}, # Dillerin birebir aynı olması istenir
}


# --- Yardımcı Fonksiyonlar ---

def extract_text_val(item: Union[str, Dict, Any]) -> str:
    """
    Gelen veri karmaşık bir yapıdaysa (Dict gibi) içindeki saf metni çeker.
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        # Olası anahtar kelimeleri kontrol et
        val = item.get('name') or item.get('dil') or item.get('Kurum') or item.get('Raw_Entry')
        if val: return str(val)
        # Hiçbiri yoksa ilk değeri al
        vals = list(item.values())
        if vals: return str(vals[0])
    return str(item) if item is not None else ""


def same_and_diff(list_a_raw: List[Any], list_b_raw: List[Any], section_name: str):
    """
    İki listeyi RapidFuzz kullanarak 'akıllıca' karşılaştırır.
    
    Args:
        list_a_raw: Birinci adayın listesi
        list_b_raw: İkinci adayın listesi
        section_name: Hangi bölümün (Eğitim, Yetenek vb.) karşılaştırıldığı
        
    Returns:
        Tuple: (ortak_elemanlar, sadece_a, sadece_b)
    """
    # 1. Veri Temizliği (Normalize et)
    cleaned_a = [clean_term(extract_text_val(x)) for x in list_a_raw if clean_term(extract_text_val(x))]
    cleaned_b = [clean_term(extract_text_val(x)) for x in list_b_raw if clean_term(extract_text_val(x))]

    # 2. Kuralı Belirle (Fuzzy vs Exact)
    rule = SECTION_MATCHING_RULES.get(section_name, {"type": "exact"})
    
    common = set()
    used_b_indices = set() # B listesinde eşleşenleri işaretle ki tekrar kullanılmasın

    # --- YÖNTEM A: FUZZY (BENZERLİK) ---
    if rule["type"] == "fuzzy":
        threshold = rule.get("threshold", 80)
        
        for item_a in cleaned_a:
            best_match_score = 0
            best_match_idx = -1
            
            # A'daki bu eleman, B'dekilerden hangisine en çok benziyor?
            for idx_b, item_b in enumerate(cleaned_b):
                if idx_b in used_b_indices: continue 
                
                # Token Sort Ratio: Kelime sırası değişse bile benzer sayar (Örn: "Java Spring" == "Spring Java")
                score = fuzz.token_sort_ratio(item_a, item_b)
                
                if score > best_match_score:
                    best_match_score = score
                    best_match_idx = idx_b
            
            # Eşik değeri geçildiyse EŞLEŞTİ say
            if best_match_score >= threshold:
                common.add(item_a) 
                used_b_indices.add(best_match_idx)

        # Kümeleri ayrıştır
        only_a = [x for x in cleaned_a if x not in common]
        only_b = [cleaned_b[i] for i in range(len(cleaned_b)) if i not in used_b_indices]

    # --- YÖNTEM B: EXACT (TAM EŞLEŞME) ---
    else:
        set_a = set(cleaned_a)
        set_b = set(cleaned_b)
        common = set_a & set_b
        only_a = list(set_a - set_b)
        only_b = list(set_b - set_a)

    return sorted(list(common)), sorted(list(only_a)), sorted(list(only_b))


def run_full_analysis(cv_file, name: str) -> Dict[str, Any]:
    """PDF dosyasını kaydeder, parse eder ve veriyi yapılandırır."""
    temp_path = os.path.join("data", f"{name}_{cv_file.name}")
    
    # Dosyayı geçici olarak kaydet
    with open(temp_path, "wb") as f:
        f.write(cv_file.getbuffer())

    # İşleme başla
    sections = parse_cv(temp_path)
    if not sections:
        return None
    return extract_structured_data(sections)


def count_for_section(data: Dict, section_key: str) -> int:
    """Verilen bölümdeki eleman sayısını döndürür (Tablo için)."""
    v = data.get(section_key)
    if v is None: return 0
    if isinstance(v, (str, list, dict)): return len(v)
    return 1


# --- Ana Uygulama Mantığı ---

def main():
    st.title("👨‍💻 CV Karşılaştırma ve Değerlendirme Sistemi")
    st.subheader("Birden fazla CV yükleyip karşılaştırabilirsiniz.")

    # 1. Dosya Yükleme Alanı
    num_cvs = st.slider("Kaç CV yüklenecek? (En az 2, en fazla 20)", min_value=2, max_value=20, value=2)

    uploaded_files = []
    cols = st.columns(2)
    for i in range(num_cvs):
        col = cols[i % 2]
        with col:
            uploaded = st.file_uploader(f"CV {i+1} Dosyasını Yükleyin (PDF)", type=["pdf"], key=f"cv_uploader_{i}")
            uploaded_files.append(uploaded)

    uploaded_present = [f for f in uploaded_files if f is not None]

    # 2. Analiz Başlatma Kontrolü
    if len(uploaded_present) < 2:
        st.info("Lütfen en az 2 adet CV yükleyin.")
        return

    if st.button("🚀 Karşılaştırmayı Başlat", type="primary"):
        with st.spinner("CV'ler parse ediliyor ve analiz ediliyor..."):
            
            # Veri Hazırlama
            data_list = []
            paired = [] # (Etiket, DosyaAdı, GörünenAd, Veri)
            
            for idx, f in enumerate(uploaded_present):
                label = chr(65 + idx) # A, B, C...
                d = run_full_analysis(f, label)
                if d:
                    display_name = os.path.splitext(f.name)[0]
                    data_list.append(d)
                    paired.append((label, f.name, display_name, d))

            if len(paired) < 2:
                st.error("Yüklenen dosyalardan en az iki tanesi okunabilir olmalı.")
                return

            # İkili Karşılaştırmalar (Pairwise Comparison)
            comparisons = []
            n = len(paired)
            for i in range(n):
                for j in range(i + 1, n):
                    lbl_i, _, disp_i, data_i = paired[i]
                    lbl_j, _, disp_j, data_j = paired[j]
                    
                    # Motoru Çalıştır
                    total_score, section_scores = compare_cv_data(data_i, data_j)
                    report_lines = generate_report(data_i, data_j, total_score, section_scores)
                    
                    comparisons.append({
                        "label": f"{disp_i} vs {disp_j}",
                        "total_score": total_score,
                        "section_scores": section_scores,
                        "report": report_lines,
                        "p1": (disp_i, data_i),
                        "p2": (disp_j, data_j)
                    })

            # --- 3. Skor Tablosu (Görselleştirme) ---
            agg_scores = {}
            # Tüm karşılaştırmalardaki bölüm başlıklarını topla
            all_sections = set().union(*[c["section_scores"].keys() for c in comparisons])
            
            # Tablo için özel sıralama
            ordered_keys = ['DENEYİM', 'YETENEKLER', 'TEKNİK_BECERİLER', 'EĞİTİM', 'YABANCI_DİL', 'SERTİFİKALAR', 'KURSLAR']
            final_keys = [k for k in ordered_keys if k in all_sections] + [k for k in all_sections if k not in ordered_keys]
            if 'ÖZET' in final_keys: final_keys.remove('ÖZET'); final_keys.append('ÖZET')

            # Her bölümün ortalama skorunu hesapla
            for section in final_keys:
                vals = [c["section_scores"].get(section, 0.0) for c in comparisons]
                agg_scores[section] = sum(vals) / len(vals) if vals else 0.0

            # Tablo Satırlarını Oluştur
            rows = []
            for section in final_keys:
                row = {'Alan': section, 'Benzerlik Skoru': f"% {agg_scores.get(section,0.0)*100:.1f}"}
                # Her adayın o alandaki eleman sayısını ekle
                for _, _, display, data in paired:
                    row[f"{display} Öğeleri"] = count_for_section(data, section)
                rows.append(row)

            # Tabloyu Göster
            scores_df = pd.DataFrame(rows)
            st.table(scores_df)

            # --- 4. Sonuçlar ve Detaylar ---
            st.header("✅ Analiz Tamamlandı")
            
            # Genel Ortalama
            all_totals = [c["total_score"] for c in comparisons]
            avg_total = sum(all_totals) / len(all_totals) if all_totals else 0.0
            st.metric(label="Genel Benzerlik Ortalaması", value=f"% {avg_total*100:.1f}")
            st.markdown("---")

            st.subheader("İK Uzmanı Raporları ve Detaylar")

            for comp in comparisons:
                with st.expander(comp["label"], expanded=False):
                    # Rapor
                    st.write("**📝 İK Uzmanı Değerlendirmesi**")
                    for line in comp["report"]:
                        st.write(line)
                    st.markdown("---")
                    
                    # Detaylı 'Aynı/Farklı' Listeleri
                    st.write("**🔍 Ortak ve Farklı Özellikler**")
                    disp_i, data_i = comp["p1"]
                    disp_j, data_j = comp["p2"]
                    
                    list_keys = ["YETENEKLER", "TEKNİK_BECERİLER", "PROJELER", "SERTİFİKALAR", "KURSLAR", "KİŞİSEL_BECERİLER", "YABANCI_DİL"]
                    
                    for key in list_keys:
                        raw_a = data_i.get(key, [])
                        raw_b = data_j.get(key, [])
                        common, only_a, only_b = same_and_diff(raw_a, raw_b, key)
                        
                        st.markdown(f"**{key}**")
                        st.write(f"✅ Ortak: {', '.join(common) if common else 'Yok'}")
                        st.write(f"🔹 {disp_i}: {', '.join(only_a) if only_a else 'Yok'}")
                        st.write(f"🔸 {disp_j}: {', '.join(only_b) if only_b else 'Yok'}")
                        st.markdown("")

            # Tüm Sertifikalar/Referanslar (Filtresiz Liste)
            for title, key in [("Tüm Kurslar / Sertifikalar", ["SERTİFİKALAR", "KURSLAR"]), ("Tüm Referanslar", ["REFERANSLAR"])]:
                all_items = []
                for _, _, display, data in paired:
                    # Key liste mi tek mi kontrol et
                    keys_to_check = key if isinstance(key, list) else [key]
                    for k in keys_to_check:
                        for item in data.get(k, []):
                            txt = extract_text_val(item)
                            if txt: all_items.append((display, txt))
                
                with st.expander(title, expanded=False):
                    if all_items:
                        for disp, txt in all_items:
                            st.write(f"**{disp}:** {txt}")
                    else:
                        st.write("Veri bulunamadı.")

if __name__ == "__main__":
    main()