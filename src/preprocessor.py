import re

class TextPreprocessor:
    @staticmethod
    def clean_text(text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s\-_]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @staticmethod
    def tokenize(text: str) -> list:
        cleaned = TextPreprocessor.clean_text(text)
        return [token for token in cleaned.split() if len(token) >= 2]
