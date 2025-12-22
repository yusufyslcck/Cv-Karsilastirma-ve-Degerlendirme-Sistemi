"""CV karşılaştırma ve semantik benzerlik hesaplama modülü."""

from typing import Dict, Any, Tuple, List
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

try:
    SEMANTIC_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    print("SBERT modeli başarıyla yüklendi.")
except Exception as e:
    print(f"HATA: Sentence Transformer yuklenemedi. Lutfen 'pip install sentence-transformers' komutunu calistirin. Hata: {e}")
    SEMANTIC_MODEL = None

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """İki metin arasındaki semantik benzerliği hesaplar."""
    if SEMANTIC_MODEL is None or not text1 or not text2:
        return 0.0
    
    embeddings = SEMANTIC_MODEL.encode([text1, text2])
    score_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
    score = score_matrix[0][0]
    return float(max(0.0, score))

def compare_cv_data(data_a: Dict[str, Any], data_b: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """İki CV'yi karşılaştırır ve benzerlik skorları üretir."""
    WEIGHTS = {
        "DENEYİM": 0.35,
        "YETENEKLER": 0.25,
        "TEKNİK_BECERİLER": 0.15,
        "EĞİTİM": 0.10,
        "ÖZET": 0.05,
        "YABANCI_DİL": 0.03,
        "KURSLAR": 0.03,
        "SERTİFİKALAR": 0.02,
        "KİŞİSEL_BECERİLER": 0.01,
        "REFERANSLAR": 0.01
    }
    
    section_scores = {}
    total_score = 0.0

    skills_a = set(data_a.get("YETENEKLER", []))
    skills_b = set(data_b.get("YETENEKLER", []))
    union_len = len(skills_a.union(skills_b))
    skill_score = len(skills_a.intersection(skills_b)) / union_len if union_len > 0 else 0.0
    section_scores["YETENEKLER"] = skill_score

    tech_a = set(data_a.get("TEKNİK_BECERİLER", []))
    tech_b = set(data_b.get("TEKNİK_BECERİLER", []))
    tech_union = len(tech_a.union(tech_b))
    tech_score = len(tech_a.intersection(tech_b)) / tech_union if tech_union > 0 else 0.0
    section_scores["TEKNİK_BECERİLER"] = tech_score

    exp_score = calculate_semantic_similarity(
        json.dumps(data_a.get("DENEYİM", [])), 
        json.dumps(data_b.get("DENEYİM", []))
    )
    section_scores["DENEYİM"] = exp_score
    
    edu_score = calculate_semantic_similarity(
        json.dumps(data_a.get("EĞİTİM", [])),
        json.dumps(data_b.get("EĞİTİM", []))
    )
    section_scores["EĞİTİM"] = edu_score

    summary_score = calculate_semantic_similarity(data_a.get("ÖZET", ""), data_b.get("ÖZET", ""))
    section_scores["ÖZET"] = summary_score
    
    langs_a = data_a.get("YABANCI_DİL", [])
    langs_b = data_b.get("YABANCI_DİL", [])
    try:
        names_a = set([l.get("dil", l).lower() if isinstance(l, dict) else str(l).lower() for l in langs_a])
        names_b = set([l.get("dil", l).lower() if isinstance(l, dict) else str(l).lower() for l in langs_b])
        lang_union = len(names_a.union(names_b))
        lang_score = len(names_a.intersection(names_b)) / lang_union if lang_union > 0 else 0.0
    except Exception:
        lang_score = calculate_semantic_similarity(json.dumps(langs_a), json.dumps(langs_b))
    section_scores["YABANCI_DİL"] = lang_score

    cert_score = calculate_semantic_similarity(json.dumps(data_a.get("SERTİFİKALAR", [])), json.dumps(data_b.get("SERTİFİKALAR", [])))
    section_scores["SERTİFİKALAR"] = cert_score

    courses_score = calculate_semantic_similarity(json.dumps(data_a.get("KURSLAR", [])), json.dumps(data_b.get("KURSLAR", [])))
    section_scores["KURSLAR"] = courses_score

    personal_score = calculate_semantic_similarity(json.dumps(data_a.get("KİŞİSEL_BECERİLER", [])), json.dumps(data_b.get("KİŞİSEL_BECERİLER", [])))
    section_scores["KİŞİSEL_BECERİLER"] = personal_score

    projects_score = calculate_semantic_similarity(json.dumps(data_a.get("PROJELER", [])), json.dumps(data_b.get("PROJELER", [])))
    section_scores["PROJELER"] = projects_score

    refs_score = calculate_semantic_similarity(json.dumps(data_a.get("REFERANSLAR", [])), json.dumps(data_b.get("REFERANSLAR", [])))
    section_scores["REFERANSLAR"] = refs_score
    # 3. Ağırlıklı Toplam Skoru Hesaplama
    for section, weight in WEIGHTS.items():
        if section in section_scores:
            total_score += section_scores[section] * weight
            
    return round(total_score, 3), section_scores

# -------------------------- RAPORLAMA --------------------------

def generate_report(data_a: Dict[str, Any], data_b: Dict[str, Any], total_score: float, section_scores: Dict[str, float]) -> List[str]:
    """
    İnsan kaynaklarına avantaj/dezavantaj raporu özeti oluşturur [cite: Plan_CV_Karsilastirma_ve_Degerlendirme_Sistemi.docx, source 5, 34].
    """
    report = [f"--- Karşılaştırma Raporu (Genel Skor: {total_score * 100:.1f}%) ---"]
    
    # Yorumlama
    if total_score > 0.75:
        report.append("-> Özet: Adaylar Yüksek Uyumlu. Semantik benzerlik, rollerin mükemmel örtüştüğünü gösteriyor.")
    elif total_score > 0.5:
        report.append("-> Özet: Adaylar Orta Uyumlu. Temel bilgi ve deneyim alanları örtüşüyor, ancak uzmanlıklar farklı.")
    else:
        report.append("-> Özet: Adaylar Düşük Uyumlu. Rol beklentisi netleştirilmeli veya uzmanlık alanları tamamen farklı.")

    # Avantaj/Dezavantaj Tespiti [cite: Plan_CV_Karsilastirma_ve_Degerlendirme_Sistemi.docx, source 34]
    
    # Deneyim
    exp_score = section_scores.get('DENEYİM', 0.0)
    report.append(f"-> DENEYİM UYUMU: {exp_score:.2f} (Semantik Örtüşme {exp_score * 100:.1f}%).")
    
    # Yetenekler Farkı
    skills_a = set(data_a.get("YETENEKLER", []))
    skills_b = set(data_b.get("YETENEKLER", []))
    
    unique_to_a = skills_a - skills_b
    unique_to_b = skills_b - skills_a
    
    if unique_to_a:
        report.append(f"-> AVANTAJ Aday A (Benzersiz Yetenek): {', '.join(list(unique_to_a)[:2])} ve fazlası.")
    if unique_to_b:
        report.append(f"-> AVANTAJ Aday B (Benzersiz Yetenek): {', '.join(list(unique_to_b)[:2])} ve fazlası.")
        
    return report