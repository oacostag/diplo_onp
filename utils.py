import re
import unicodedata
import nltk
from nltk.corpus import stopwords
import numpy as np

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def clean_text(text, pattern="[^a-zA-Z ]"):
    cleaned_text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore')
    cleaned_text = re.sub(pattern, "", cleaned_text.decode("utf-8"), flags=re.UNICODE)
    cleaned_text = u' '.join(cleaned_text.lower().split())
    return cleaned_text

def get_stop_words():
    stop_words = stopwords.words("english") + ["url", "seealso", "mashable", "http", "ha", "hi", "will", "new", "one", "ad", "say", "said", "people", "time", "company", "make"]
    stop_words = [clean_text(word) for word in stop_words]
    return stop_words

def remove_urls(text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.#&+])+', text)
    for url in urls:
        text = text.replace(url, "")
    return text, urls

def process_text_and_title(text, title):
    text_no_urls, urls = remove_urls(text)
    
    stop_words = get_stop_words()
    text_cleaned = clean_text(text_no_urls)
    text_cleaned = " ".join([word for word in text_cleaned.split() if word not in stop_words])
    
    title_cleaned = title.replace("\n", " ")
    title_cleaned = clean_text(title_cleaned)
    title_cleaned = " ".join([word for word in title_cleaned.split() if word not in stop_words])
    
    return text_cleaned, title_cleaned, urls
