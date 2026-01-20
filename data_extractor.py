"""
Data Extractor Modülü
Bu modül, metin haline getirilmiş CV verilerinden yapılandırılmış bilgi
(Deneyim, Eğitim, Yetenekler vb.) çıkarmak için spaCy ve Regex kullanır.
"""

import re
import spacy
from typing import Dict, List, Any, Optional
from collections import defaultdict

# --- Model Yapılandırması ---
CUSTOM_NER_MODEL_NAME = "en_core_web_sm"

try:
    nlp = spacy.load(CUSTOM_NER_MODEL_NAME)
    print(f"✅ NLP: '{CUSTOM_NER_MODEL_NAME}' modeli başarıyla yüklendi.")
except OSError:
    print(f"⚠️ HATA: '{CUSTOM_NER_MODEL_NAME}' bulunamadı. Temel modele geçiliyor.")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✅ NLP: Yedek model 'en_core_web_sm' yüklendi.")
    except Exception as e:
        print(f"❌ KRİTİK HATA: Hiçbir spaCy modeli yüklenemedi! Hata: {e}")
        nlp = None


# --- Regex Sabitleri (Performans İçin Derlendi) ---
# Dil seviyelerini yakalamak için (Advanced, B2, Orta vb.)
LEVEL_PATTERN = re.compile(
    r"(advanced|intermediate|basic|fluent|beginner|native|"
    r"ileri|orta|başlangıç|başlangic|çok iyi|iyi|a1|a2|b1|b2|c1|c2)", 
    re.IGNORECASE
)

# Tarihleri yakalamak için (1990-2099 arası yıllar veya 'Present/Devam')
DATE_PATTERN = re.compile(r'(?:19|20)\d{2}|Present|Devam|Günümüz', re.IGNORECASE)


# --- Yardımcı Fonksiyonlar ---

def extract_list_from_text(text: str) -> List[str]:
    """
    Virgül, noktalı virgül veya yeni satırla ayrılmış metinleri temiz bir listeye çevirir.
    Örn: "Python, Java; C++" -> ["Python", "Java", "C++"]
    """
    if not text:
        return []
    
    # Metni yaygın ayırıcılara göre böl
    items = [s.strip() for s in re.split(r'[,;•\n\t]', text) if len(s.strip()) > 1]
    
    # Tekrarları önlemek için set kullanıp listeye çeviriyoruz
    return sorted(list(set([s for s in items if s])))


def extract_languages(text: str) -> List[Dict[str, str]]:
    """
    Metin içindeki yabancı dilleri ve varsa seviyelerini çıkarır.
    Örn: "İngilizce (İleri)" -> {'dil': 'İngilizce', 'seviyesi': 'İleri'}
    """
    if not text: return []
    
    parts = [p.strip() for p in re.split(r'[;,\n\t•]', text) if p.strip()]
    langs = []
    
    for p in parts:
        if len(p) < 2: continue 
        
        match = LEVEL_PATTERN.search(p)
        if match:
            level = match.group(0)
            # Seviye bilgisini metinden çıkarıp sadece dil ismini bırak
            name = LEVEL_PATTERN.sub('', p).strip(' -,:;()')
            if not name: name = p # Eğer isim silindiyse orijinali kullan
            langs.append({"dil": name, "seviyesi": level})
        else:
            langs.append({"dil": p, "seviyesi": ""})
            
    return langs


def extract_references(text: str) -> List[Dict[str, str]]:
    """
    Referans bölümünü analiz eder.
    'İstek üzerine verilecektir' gibi boş ibareleri eler.
    """
    if not text: return []
    
    # Yaygın 'boş' referans cümlelerini kontrol et
    if len(text) < 50 and ("istek" in text.lower() or "request" in text.lower()):
        return []

    # Satır satır analiz
    lines = [line.strip() for line in text.split('\n') if len(line.strip()) > 3]
    refs = []
    
    for line in lines:
        # Başlık tekrarını ve e-posta/telefon satırlarını atla (basit filtre)
        if "referans" in line.lower(): continue
        
        # Basit isim kontrolü (En az iki kelime olmalı)
        if len(line.split()) >= 2 and len(line) < 50:
             refs.append({"name": line, "raw": line})
             
    return refs


# --- Ana Yapılandırma Fonksiyonları ---

def extract_experience_details(experience_text: str) -> List[Dict[str, str]]:
    """
    Deneyim metnini iş pozisyonlarına (bloklara) ayırır ve detaylandırır.
    NLP ve Regex kullanarak tarih ve kurum ismini bulmaya çalışır.
    """
    if not nlp or not experience_text: return []

    experiences = []
    
    # Yöntem 1: Çift satır boşluğu ile ayırma (Standart format)
    entries = re.split(r'\n\s*\n', experience_text.strip())
    
    # Yöntem 2: Eğer tek blok geldiyse, tarih desenlerine göre bölmeyi dene
    if len(entries) <= 1:
        lines = experience_text.split('\n')
        current_block = []
        entries = [] # Reset
        
        for line in lines:
            # Satırda tarih varsa ve blok zaten doluysa -> Yeni Blok Başlangıcı
            if DATE_PATTERN.search(line) and len(current_block) > 0:
                entries.append("\n".join(current_block))
                current_block = [line]
            else:
                current_block.append(line)
        
        if current_block:
            entries.append("\n".join(current_block))
            
    # Eğer hala bölünemediyse orijinal metni tek parça al
    if not entries:
        entries = [experience_text]

    # Blokları işle ve yapılandır
    for entry in entries:
        if len(entry.strip()) < 5: continue 

        details = defaultdict(str)
        
        # NLP ile Varlık İsmi Tanıma (NER)
        doc = nlp(entry.strip())
        for ent in doc.ents:
            if ent.label_ == "DATE":
                details["Tarih"] += ent.text + " "
            elif ent.label_ == "ORG": # Organizasyon/Şirket
                if not details["Kurum"]: details["Kurum"] = ent.text
        
        # İlk satırı genelde başlık/pozisyon olarak kabul et
        first_line = entry.split('\n')[0].strip()
        details["Raw_Entry"] = first_line
        
        # Sadece anlamlı verileri ekle
        if details.get("Kurum") or details.get("Tarih") or details.get("Raw_Entry"):
            experiences.append(dict(details))
            
    return experiences


def extract_education_details(education_text: str) -> List[Dict[str, str]]:
    """
    Eğitim metnini okullara/derecelere ayırır.
    Üniversite veya Okul kelimelerini baz alarak bloklama yapar.
    """
    if not nlp or not education_text: return []

    # Yöntem 1: Çift satır
    entries = re.split(r'\n\s*\n', education_text.strip())
    
    # Yöntem 2: Anahtar kelimeye göre bölme (Üniversite, Lise, School vb.)
    if len(entries) <= 1:
        lines = education_text.split('\n')
        current_block = []
        entries = [] 
        
        keywords = ["niversi", "School", "Lise", "Bachelor", "Master", "PhD", "Lisans"]
        
        for line in lines:
            # Satırda anahtar kelime varsa -> Yeni Blok
            if any(k in line for k in keywords) and len(current_block) > 0:
                 entries.append("\n".join(current_block))
                 current_block = [line]
            else:
                 current_block.append(line)
        
        if current_block: entries.append("\n".join(current_block))

    education_list = []
    for entry in entries:
        if len(entry.strip()) < 5: continue
        
        details = defaultdict(str)
        doc = nlp(entry.strip())
        
        for ent in doc.ents:
             if ent.label_ == "DATE": details["Tarih"] += ent.text + " "
             elif ent.label_ == "ORG": details["Kurum"] = ent.text

        details["Raw_Entry"] = entry.strip()
        if details.get("Raw_Entry"):
            education_list.append(dict(details))

    return education_list


def extract_structured_data(sections: Dict[str, str]) -> Dict[str, Any]:
    """
    Modülün ana fonksiyonu.
    Sözlük halindeki ham bölümleri alır, ilgili ayrıştırıcı fonksiyonlara gönderir
    ve son kullanıcıya sunulacak yapılandırılmış veriyi (JSON benzeri) oluşturur.
    """
    structured_data = {
        "DENEYİM": [], "EĞİTİM": [], "YETENEKLER": [], "SERTİFİKALAR": [], 
        "PROJELER": [], "TEKNİK_BECERİLER": [], "YABANCI_DİL": [],
        "KURSLAR": [], "KİŞİSEL_BECERİLER": [], "REFERANSLAR": [],
        "ÖZET": sections.get("ÖZET", sections.get("SUMMARY", ""))
    }
    
    # Karmaşık Alanlar (Eğitim ve Deneyim)
    if sections.get("DENEYİM") or sections.get("EXPERIENCE"):
        raw_exp = sections.get("DENEYİM") or sections.get("EXPERIENCE")
        structured_data["DENEYİM"] = extract_experience_details(raw_exp)
    
    if sections.get("EĞİTİM") or sections.get("EDUCATION"):
        raw_edu = sections.get("EĞİTİM") or sections.get("EDUCATION")
        structured_data["EĞİTİM"] = extract_education_details(raw_edu)

    # Liste Tabanlı Basit Alanlar
    mappings = [
        ("YETENEKLER", ["YETENEKLER", "SKILLS"]),
        ("TEKNİK_BECERİLER", ["TEKNİK BECERİLER", "TECHNICAL SKILLS"]),
        ("SERTİFİKALAR", ["SERTİFİKALAR", "CERTIFICATIONS"]),
        ("KURSLAR", ["KURSLAR", "COURSES"]),
        ("KİŞİSEL_BECERİLER", ["KİŞİSEL BECERİLER", "PERSONAL SKILLS"]),
    ]

    for target_key, source_keys in mappings:
        text = next((sections.get(k) for k in source_keys if sections.get(k)), None)
        if text:
            structured_data[target_key] = extract_list_from_text(text)

    # Özel Formatlı Alanlar
    lang_text = sections.get("YABANCI DİL") or sections.get("LANGUAGES")
    if lang_text:
        structured_data["YABANCI_DİL"] = extract_languages(lang_text)

    refs_text = sections.get("REFERANSLAR") or sections.get("REFERENCES")
    if refs_text:
        structured_data["REFERANSLAR"] = extract_references(refs_text)

    proj_text = sections.get("PROJELER") or sections.get("PROJECTS")
    if proj_text:
        # Projeler şimdilik tek bir blok olarak saklanıyor
        structured_data["PROJELER"].append({"Raw_Entry": proj_text.strip()})

    return structured_data