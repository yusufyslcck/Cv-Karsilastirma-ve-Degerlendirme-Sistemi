"""PDF dosyalarından metin çıkarma ve CV bölümlerini ayrıştırma modülü."""

import pdfplumber
import re
from typing import Dict, Optional

try:
    import easyocr
    import fitz
    import numpy as np
    print("EasyOCR modülleri yükleniyor...")
    OCR_READER = easyocr.Reader(['tr', 'en'], gpu=False, verbose=False)
    print("✅ EasyOCR hazır (Türkçe + İngilizce, PyMuPDF ile)")
except ImportError as e:
    print(f"⚠️ EasyOCR yüklenemedi: {e}")
    OCR_READER = None
except Exception as e:
    print(f"⚠️ EasyOCR başlatılamadı: {e}")
    OCR_READER = None

def extract_text_with_ocr(pdf_path: str) -> Optional[str]:
    """EasyOCR ile taranmış PDF'den metin çıkarır."""
    if not OCR_READER:
        print("❌ OCR mevcut değil")
        return None
    try:
        import fitz
        print(f"📄 OCR başlatılıyor: {pdf_path}")
        
        # PyMuPDF ile PDF'i aç
        doc = fitz.open(pdf_path)
        print(f"✅ {len(doc)} sayfa bulundu")
        
        full_text = ""
        for i, page in enumerate(doc, 1):
            print(f"  📖 Sayfa {i}/{len(doc)} okunuyor...")
            
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            if pix.n == 4:
                img_data = img_data[:, :, :3]
            result = OCR_READER.readtext(img_data, detail=0, paragraph=True)
            page_text = " ".join(result)
            full_text += page_text + "\n\n"
            print(f"  ✅ Sayfa {i}: {len(page_text)} karakter okundu")
        
        doc.close()
        
        if full_text.strip():
            print(f"✅ OCR tamamlandı: Toplam {len(full_text)} karakter")
            return full_text
        else:
            print("⚠️ OCR hiç metin bulamadı")
            return None
    except Exception as e:
        print(f"❌ OCR Hatası: {e}")
        import traceback
        traceback.print_exc()
        return None

def extract_text_from_pdf(pdf_path: str) -> Optional[str]:
    """PDF dosyasından metin çıkarır, gerekirse OCR kullanır."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n\n"
            
            if full_text and len(full_text.strip()) > 100:
                print(f"✅ pdfplumber başarılı: {len(full_text)} karakter")
                return full_text
            
            if OCR_READER:
                print(f"⚠️ PDF'de metin yetersiz ({len(full_text.strip())} karakter), OCR deneniyor...")
                ocr_text = extract_text_with_ocr(pdf_path)
                if ocr_text:
                    return ocr_text
                else:
                    print("⚠️ OCR de başarısız, mevcut metin döndürülüyor")
            
            return full_text if full_text else None
            
    except Exception as e:
        print(f"Hata: PDF okunamadı {pdf_path}. Hata: {e}")
        return None

def preprocess_text(text: str) -> str:
    """Metni temizler ve düzenler."""
    if not text:
        return ""
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def extract_sections_simple(text: str) -> Dict[str, str]:
    """CV metninden bölümleri ayırır ve yapılandırır."""
    section_titles = [
        "EĞİTİM", "Egitim", "DENEYİM", "Deneyim", "YETENEKLER", "Yetenekler",
        "TEKNİK BECERİLER", "TEKNIK BECERILER", "TEKNİK", "TECHNICAL SKILLS",
        "YABANCI DİL", "YABANCI DİLLER", "LANGUAGES", "DİL", "DIL",
        "KURSLAR", "KURS", "COURSES",
        "SERTİFİKALAR", "CERTIFICATIONS",
        "KİŞİSEL BECERİLER", "KISISEL BECERILER", "PERSONAL SKILLS",
        "REFERANSLAR", "REFERANS", "REFERENCES",
        "SKILLS", "EXPERIENCE", "EDUCATION", "SUMMARY", "ÖZET", "CONTACT", "İLETİŞİM", "PROJELER"
    ]
    
    pattern = r'\b(' + '|'.join(re.escape(title) for title in section_titles) + r')\b'
    
    parts = replit_with_content(pattern, text)
    
    sections = {}
    current_title = "GENERAL"
    
    for part in parts:
        if part.strip().upper() in [t.upper() for t in section_titles]:
            current_title = part.strip().upper()
            sections[current_title] = ""
        elif current_title in sections:
            sections[current_title] += part.strip() + " "
        else:
            sections["GENERAL"] = sections.get("GENERAL", "") + part.strip() + " "

    return {k: v.strip() for k, v in sections.items() if v.strip()}

def replit_with_content(pattern: str, text: str) -> list:
    """re.split'in yakalanan grupları dahil etme versiyonu."""
    parts = re.split(pattern, text, flags=re.IGNORECASE)
    return [p for p in parts if p and p.strip()]

def parse_cv(pdf_path: str) -> Dict[str, str]:
    """PDF'den metin çıkarır ve bölümlere ayırır."""
    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text:
        return {}
    
    print(f"\n🔍 DEBUG - Ham metin ilk 600 karakter:\n{raw_text[:600]}\n")
    print(f"🔍 DEBUG - Ham metin son 300 karakter:\n{raw_text[-300:]}\n")
    
    sections = extract_sections_simple(raw_text)
    
    print(f"🔍 DEBUG - Bulunan bölümler: {list(sections.keys())}")
    for key, value in sections.items():
        print(f"  - {key}: {len(value)} karakter (ilk 100: {value[:100]}...)")
    
    return sections