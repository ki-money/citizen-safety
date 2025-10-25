from googletrans import Translator
from deep_translator import GoogleTranslator
import time
import logging

logging.basicConfig(level=logging.INFO)

def safe_translate(text: str, src: str = 'auto', dest: str = 'en', retries: int = 3) -> str:
    if not text.strip(): return text
    for i in range(retries):
        try:
            t = Translator()
            return t.translate(text, src=src, dest=dest).text
        except Exception as e:
            logging.warning(f"googletrans fail {i+1}: {e}")
            if i < retries - 1: time.sleep(1)
            else:
                try:
                    return GoogleTranslator(source=src, target=dest).translate(text)
                except:
                    return "[Translation unavailable]"
    return "[Translation failed]"

LANG_MAP = {'en': 'en', 'sw': 'sw', 'ki': 'sw', 'kikuyu': 'sw'}

def detect_language(text: str) -> str:
    try:
        return Translator().detect(text).lang.lower()
    except:
        return 'en'

def translate_text(text: str, target_lang: str) -> str:
    target = LANG_MAP.get(target_lang.lower(), 'en')
    src = detect_language(text)
    if src in ('ki', 'kikuyu'): src = 'sw'
    if src == target: return text
    return safe_translate(text, src=src, dest=target)

def translate_report(report_dict: dict, target_lang: str) -> dict:
    return {k: translate_text(v, target_lang) for k, v in report_dict.items()}