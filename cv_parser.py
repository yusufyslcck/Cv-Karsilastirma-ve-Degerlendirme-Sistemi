"""
CV Parser Modülü (Hata Odaklı Sessiz Mod)
Başarılı işlemleri sessizce yapar, sadece hataları raporlar.
CV içeriklerini terminale basmaz.
"""

import pdfplumber
import re
import traceback
from typing import Dict, Optional, List

# --- Opsiyonel Kütüphaneler ve OCR Hazırlığı ---
try:
    import easyocr
    import fitz  
    import numpy as np
    
    # Başlangıçta sadece bir kere bilgi verir, sürekli yazmaz.
    OCR_READER = easyocr.Reader(['tr', 'en'], gpu=False, verbose=False)

except ImportError as e:
    print(f"⚠️ UYARI: EasyOCR modülü eksik. Sadece metin tabanlı PDF'ler okunabilir.")
    OCR_READER = None
except Exception as e:
    print(f"❌ KRİTİK: OCR motoru başlatılamadı: {e}")
    OCR_READER = None


# --- Sabitler: Tanınan Bölüm Başlıkları ---
KNOWN_SECTION_HEADERS = [
    "EĞİTİM", "Egitim", "EDUCATION", 
    "DENEYİM", "Deneyim", "EXPERIENCE", "İŞ DENEYİMİ",
    "YETENEKLER", "Yetenekler", "SKILLS", "YETKİNLİKLER",
    "TEKNİK BECERİLER", "TEKNIK BECERILER", "TEKNİK", "TECHNICAL SKILLS",
    "YABANCI DİL", "YABANCI DİLLER", "LANGUAGES", "DİL", "DIL",
    "KURSLAR", "KURS", "COURSES",
    "SERTİFİKALAR", "CERTIFICATIONS", "SERTIFIKALAR",
    "KİŞİSEL BECERİLER", "KISISEL BECERILER", "PERSONAL SKILLS",
    "REFERANSLAR", "REFERANS", "REFERENCES",
    "SUMMARY", "ÖZET", "PROFIL", "PROFILE",
    "CONTACT", "İLETİŞİM", "ILETISIM",
    "PROJELER", "PROJECTS"
]


# --- Yardımcı Fonksiyonlar ---

def preprocess_text(text: str) -> str:
    """Çıkarılan ham metni temizler."""
    if not text:
        return ""
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def custom_split(pattern: str, text: str) -> List[str]:
    """re.split fonksiyonunun özelleştirilmiş hali."""
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [p for p in parts if p and p.strip()]


# --- Ana İşlem Fonksiyonları ---

def extract_text_with_ocr(pdf_path: str) -> Optional[str]:
    """EasyOCR ve PyMuPDF kullanarak metin çıkarır."""
    if not OCR_READER:
        print("❌ HATA: OCR gerekli ama OCR motoru yüklü değil.")
        return None
        
    try:
        # İşlem başlıyor 
        doc = fitz.open(pdf_path)
        full_text = ""
        
        for page in doc:
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            if pix.n == 4:
                img_data = img_data[:, :, :3]
            
            # detail=0 ile sadece metin
            result = OCR_READER.readtext(img_data, detail=0, paragraph=True)
            page_text = " ".join(result)
            full_text += page_text + "\n\n"
        
        doc.close()
        
        if not full_text.strip():
            print(f"⚠️ UYARI: OCR çalıştı ama {pdf_path} dosyasında metin bulunamadı.")
            return None
            
        return full_text

    except Exception as e:
        # BURASI ÖNEMLİ: Hata olursa terminalde görünecek
        print(f"❌ OCR HATASI ({pdf_path}): {e}")
        traceback.print_exc() # Hatanın detayını da basar
        return None


def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """PDF dosyasından metin çıkarmayı dener."""
    try:
        # Yöntem 1: Standart Metin Çıkarma
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
            
            # Başarılıysa sessizce dön
            if full_text and len(full_text.strip()) > 100:
                return full_text
            
            # Yöntem 2: OCR
            if OCR_READER:
                ocr_text = extract_text_with_ocr(pdf_path)
                if ocr_text:
                    return ocr_text
            
            return full_text if full_text else None
            
    except Exception as e:
        # Hata olursa basar
        print(f"❌ PDF OKUMA HATASI ({pdf_path}): {e}")
        return None


def extract_sections_simple(text: str) -> Dict[str, str]:
    """Ham metni regex kullanarak mantıksal bölümlere ayırır."""
    pattern = r'\b(' + '|'.join(re.escape(title) for title in KNOWN_SECTION_HEADERS) + r')\b'
    parts = custom_split(pattern, text)
    
    sections = {}
    current_title = "GENERAL"
    
    for part in parts:
        part_clean = part.strip()
        if part_clean.upper() in [t.upper() for t in KNOWN_SECTION_HEADERS]:
            current_title = part_clean.upper()
            if current_title not in sections:
                sections[current_title] = ""
        elif current_title in sections:
            sections[current_title] += part_clean + " "
        else:
            sections["GENERAL"] = sections.get("GENERAL", "") + part_clean + " "

    return {k: v.strip() for k, v in sections.items() if v.strip()}


def parse_cv(pdf_path: str) -> Dict[str, str]:
    """Modülün ana giriş noktasıdır."""
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        return {}
    
    return extract_sections_simple(raw_text)