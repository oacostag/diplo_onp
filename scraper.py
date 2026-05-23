import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def scrape(url):
    req = requests.get(url)
    return url, req.text

def get_data(soup):
    title = soup.find_all("h1")[0].text if len(soup.find_all("h1")) > 0 else ''
    subtitles = soup.find_all("h2")
    channel = subtitles[0].text if len(subtitles) > 0 else ''
    images = len(soup.find_all("img"))
    videos = len(soup.find_all("iframe"))
    paragraphs = soup.find_all("p")
    text = "\n".join([x.text for x in paragraphs])
    return title, len(subtitles), channel, images, videos, len(paragraphs), text

def text_features(url, text):
    sid = SentimentIntensityAnalyzer()
    sentences = sent_tokenize(text)
    if not sentences:
        return pd.DataFrame()
    polarities = pd.DataFrame(map(sid.polarity_scores, sentences))
    words = pd.DataFrame(map(len, map(word_tokenize, sentences)))
    sentences_data = polarities.join(words).rename(columns={0: "n_words"})
    sentences_data["url"] = url
    return sentences_data
