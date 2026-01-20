"""
Comparison Engine Modülü
Bu modül, iki CV verisi arasındaki benzerliği hesaplamak için
semantik analiz (SBERT), fuzzy matching ve ağırlıklı puanlama algoritmalarını kullanır.
"""

import json
import re
import traceback
from typing import Dict, Any, Tuple, List, Set, Optional

# --- Gerekli Kütüphaneler ---
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    raise ImportError(f"Gerekli kütüphaneler eksik: {e}. Lütfen requirements.txt'yi kontrol edin.")

# --- Model Yükleme (Singleton Yaklaşımı) ---
try:
    # Çok dilli model: Türkçe ve İngilizce için optimize edilmiştir.
    print("⏳ SBERT Modeli yükleniyor (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)...")
    SEMANTIC_MODEL = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    print("✅ SBERT Modeli başarıyla yüklendi.")
except Exception as e:
    print(f"⚠️ UYARI: SBERT yüklenemedi. Semantik analiz devre dışı kalacak. Hata: {e}")
    SEMANTIC_MODEL = None


# --- Sabitler ve Ayarlar ---

# Bölümlerin genel skora katkı oranları (Toplam 1.0 olmalı)
SECTION_WEIGHTS = {
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

# Temizlenecek gereksiz kelimeler (Stopwords)
STOPWORDS = {
    "ve", "ile", "için", "bir", "bu", "şu", "o", "de", "da", "ki", "mi", "mı",
    "olarak", "olan", "ilgili", "gibi", "üzere", "üzerinde", "tarafından", "birlikte",
    "yaptım", "ettim", "sağladım", "geliştirdim", "çalıştım", "bulundum",
    "uyguladım", "yönettim", "belirledim", "oluşturdum", "güçlendirdim",
    "hazırladım", "tasarladım", "destek", "görev", "aldım", "proje", "ekibi",
    "deneyim", "yıl", "ay", "kullanımı", "bilgisi", "hakkında"
}

# Temizlenecek dil seviyesi ifadeleri
LANGUAGE_LEVELS = [
    "a1", "a2", "b1", "b2", "c1", "c2", 
    "başlangıç", "orta", "ileri", "native", "fluent", 
    "beginner", "intermediate", "advanced", "ana dil", "seviye", "level"
]


# --- Yardımcı Metin İşleme Fonksiyonları ---

def tr_lower(text: str) -> str:
    """Türkçe karakter uyumlu küçük harfe çevirme (I -> ı, İ -> i)."""
    if not text: return ""
    return text.replace("İ", "i").replace("I", "ı").lower()


def clean_stopwords(text: str) -> str:
    """Metni temizler: Özel karakterleri siler, stopwords'ü atar."""
    if not text: return ""
    text = tr_lower(text)
    
    # Özel karakterleri ve sembolleri (■, ▪, ● vb.) temizle
    # [^\w\s] -> Harf ve sayı olmayan her şeyi sil
    text = re.sub(r'[^\w\s]', ' ', text)
    
    words = text.split()
    cleaned_words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return " ".join(cleaned_words)


def clean_term(text: str, is_language: bool = False) -> str:
    """
    Tekil terimleri temizler.
    Liste elemanlarını (Yetenekler vb.) normalize etmek için kullanılır.
    """
    text = tr_lower(text)
    
    # Gereksiz parantez ve tırnakları temizle
    text = text.replace("[", "").replace("]", "").replace("'", "").replace('"', "")
    
    # Sadece harf ve rakam kalsın
    text = re.sub(r'[^\w\s]', ' ', text)
    
    if is_language:
        for level in LANGUAGE_LEVELS:
            text = text.replace(level, "")
            
    return text.strip()


def normalize_set(data_list: Any, field_type: str = "") -> Set[str]:
    """
    Gelen veriyi (List, String, Dict) standart bir Python set'ine (küme) dönüştürür.
    Bu işlem karşılaştırma öncesi veriyi atomik parçalara ayırır.
    """
    normalized = set()
    if not data_list:
        return normalized

    # Girdiyi liste formatına zorla
    raw_items = []
    if isinstance(data_list, str):
        raw_items = [data_list]
    elif isinstance(data_list, list):
        raw_items = data_list
    else:
        raw_items = [str(data_list)]

    is_lang = (field_type == "YABANCI_DİL")

    for item in raw_items:
        text_val = ""
        # Eğer öğe bir sözlükse (Dict), anlamlı değeri bulmaya çalış
        if isinstance(item, dict):
            # Olası anahtar kelimeler
            possible_keys = ['dil', 'name', 'yetenek', 'Kurum', 'school', 'title', 'company']
            for key in possible_keys:
                # Hem normal hem küçük harfli halini dene
                if key in item:
                    text_val = str(item[key]); break
                if key.lower() in item: 
                    text_val = str(item[key.lower()]); break
            
            # Anahtar bulunamazsa tüm değerleri birleştir
            if not text_val: 
                text_val = " ".join([str(v) for v in item.values()])
        else:
            text_val = str(item)

        # PARÇALAYICI (Splitter):
        # Virgül, noktalı virgül, yeni satır, tab ve mermi işaretlerinden böl.
        splitted_items = re.split(r'[,\n;\•\t■▪●]+|\s{2,}', text_val)
        
        for sub_item in splitted_items:
            cleaned = clean_term(sub_item, is_language=is_lang)
            
            # 'C' ve 'R' dillerini koru, diğer tek harflileri at
            if cleaned:
                if len(cleaned) > 1 or cleaned in ['c', 'r']:
                    normalized.add(cleaned)
            
    return normalized


# --- Ana Karşılaştırma Mantığı ---

def find_fuzzy_commons(set_a: Set[str], set_b: Set[str], threshold: float = 0.80) -> Set[str]:
    """
    İki küme arasındaki ortak elemanları bulur.
    3 Aşamalı Kontrol Yapar:
    1. Birebir Eşleşme (Exact Match)
    2. Kapsama Kontrolü (Containment) - Örn: "Excel" vs "Microsoft Excel"
    3. Semantik Benzerlik (SBERT) - Örn: "AI" vs "Artificial Intelligence"
    """
    if not set_a or not set_b:
        return set()
    
    # 1. Aşama: Tam Eşleşenler
    commons = set_a.intersection(set_b)
    
    # Eşleşmeyenleri ayır
    diff_a = list(set_a - commons)
    diff_b = list(set_b - commons)
    
    if not diff_a or not diff_b:
        return commons

    matched_indices_a = set()
    
    # 2. Aşama: Kapsama Kontrolü (Substring Check)
    for i, term_a in enumerate(diff_a):
        for term_b in diff_b:
            if term_a in term_b or term_b in term_a:
                commons.add(term_a) # Kısa olanı ekle (Genelde daha temizdir)
                matched_indices_a.add(i)
                break 

    # 3. Aşama: SBERT ile Vektör Benzerliği
    # Sadece hala eşleşmemiş olanlara bakılır
    remaining_a = [val for i, val in enumerate(diff_a) if i not in matched_indices_a]
    
    if remaining_a and SEMANTIC_MODEL is not None:
        embeddings_a = SEMANTIC_MODEL.encode(remaining_a)
        embeddings_b = SEMANTIC_MODEL.encode(diff_b)
        
        # Kosinüs benzerlik matrisi
        similarity_matrix = cosine_similarity(embeddings_a, embeddings_b)
        
        for i, term_a in enumerate(remaining_a):
            # En iyi eşleşmeyi bul
            best_match_idx = np.argmax(similarity_matrix[i])
            best_score = similarity_matrix[i][best_match_idx]
            
            if best_score > threshold:
                commons.add(term_a)

    return commons


def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """
    İki uzun metin bloğu (Örn: Deneyimler) arasındaki anlamsal benzerliği hesaplar.
    Düşük skorları cezalandırmak için skorun küpünü (x^3) alır.
    """
    if SEMANTIC_MODEL is None or not text1 or not text2: return 0.0
    
    clean1 = clean_stopwords(text1)
    clean2 = clean_stopwords(text2)
    
    if not clean1 or not clean2: return 0.0

    embeddings = SEMANTIC_MODEL.encode([clean1, clean2])
    score_matrix = cosine_similarity([embeddings[0]], [embeddings[1]])
    
    raw_score = float(max(0.0, score_matrix[0][0]))
    
    # Küpünü alarak farkı belirginleştir (0.5 -> 0.125 olur, 0.9 -> 0.729 olur)
    return raw_score ** 3 


def get_field_data(data: Dict, keys: List[str]) -> Any:
    """Verilen anahtarlara göre sözlükten esnek veri çeker (Case-insensitive)."""
    for k in keys:
        if k in data: return data[k]
        if k.lower() in data: return data[k.lower()]
        # Boşluk/Alt çizgi dönüşümlerini dene
        if k.replace("_", " ") in data: return data[k.replace("_", " ")]
        if k.lower().replace("_", " ") in data: return data[k.lower().replace("_", " ")]
    return []


def compare_cv_data(data_a: Dict[str, Any], data_b: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    """
    İki adayın tüm verilerini ağırlıklandırılmış algoritmaya göre karşılaştırır.
    
    Returns:
        (Toplam Skor, {Bölüm Bazlı Skorlar})
    """
    section_scores = {}
    total_score = 0.0
    
    # Liste Tabanlı Alanlar (Küme Benzerliği)
    list_fields = ["YETENEKLER", "TEKNİK_BECERİLER", "YABANCI_DİL", 
                   "KİŞİSEL_BECERİLER", "KURSLAR", "SERTİFİKALAR", "REFERANSLAR"]
    
    for field in list_fields:
        val_a = get_field_data(data_a, [field])
        val_b = get_field_data(data_b, [field])
        
        set_a = normalize_set(val_a, field_type=field)
        set_b = normalize_set(val_b, field_type=field)
        
        commons = find_fuzzy_commons(set_a, set_b)
        
        # Jaccard Benzerliği Benzeri Skorlama
        union_len = len(set_a.union(set_b))
        if union_len > 0:
            score = len(commons) / union_len
        else:
            score = 0.0
        section_scores[field] = score

    # Semantik Alanlar (Metin Benzerliği)
    semantic_fields = ["DENEYİM", "EĞİTİM", "ÖZET", "PROJELER"]
    for field in semantic_fields:
        val_a = get_field_data(data_a, [field])
        val_b = get_field_data(data_b, [field])
        
        # Listeleri string'e dök
        text_a = json.dumps(val_a, ensure_ascii=False) if isinstance(val_a, list) else str(val_a)
        text_b = json.dumps(val_b, ensure_ascii=False) if isinstance(val_b, list) else str(val_b)
        
        section_scores[field] = calculate_semantic_similarity(text_a, text_b)

    # Ağırlıklı Toplam Hesaplama
    for section, weight in SECTION_WEIGHTS.items():
        if section in section_scores:
            total_score += section_scores[section] * weight
            
    return round(total_score, 3), section_scores


# --- Raporlama Fonksiyonları ---

def generate_report(data_a: Dict[str, Any], data_b: Dict[str, Any], total_score: float, section_scores: Dict[str, float]) -> List[str]:
    """
    Karşılaştırma sonucuna göre okunabilir bir metin raporu oluşturur.
    """
    report = [f"--- Karşılaştırma Raporu (Genel Skor: {total_score * 100:.1f}%) ---"]
    
    # Genel Yorum
    if total_score > 0.75: report.append("-> Özet: Adaylar Yüksek Uyumlu.")
    elif total_score > 0.5: report.append("-> Özet: Adaylar Orta Uyumlu.")
    else: report.append("-> Özet: Adaylar Düşük Uyumlu.")

    report.append(f"-> DENEYİM UYUMU: %{section_scores.get('DENEYİM', 0)*100:.1f}")

    # Ortak Özelliklerin Listelenmesi
    check_fields = [
        ("YETENEKLER", "ORTAK YETENEKLER"),
        ("TEKNİK_BECERİLER", "ORTAK TEKNİK BECERİLER"),
        ("YABANCI_DİL", "ORTAK DİLLER"),
        ("KİŞİSEL_BECERİLER", "ORTAK KİŞİSEL BECERİLER"),
        ("SERTİFİKALAR", "ORTAK SERTİFİKALAR"),
        ("KURSLAR", "ORTAK KURSLAR")
    ]
    
    for field_key, title in check_fields:
        val_a = get_field_data(data_a, [field_key])
        val_b = get_field_data(data_b, [field_key])
        
        set_a = normalize_set(val_a, field_type=field_key)
        set_b = normalize_set(val_b, field_type=field_key)
        
        common = find_fuzzy_commons(set_a, set_b)
        
        if common:
            # Okunabilirlik için baş harfleri büyüt
            common_list = [i.title() for i in list(common)]
            # Çok uzunsa kısalt
            display_str = ', '.join(common_list[:10])
            if len(common_list) > 10: display_str += "..."
            report.append(f"-> {title}: {display_str}")
            
    return report