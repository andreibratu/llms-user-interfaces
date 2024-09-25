from concurrent.futures import ThreadPoolExecutor

import requests
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.document_transformers import Html2TextTransformer
from readability import Document as ReadabilityDocument
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer

from src.configuration import APP_CONFIG

_OPTIONS = Options()
_OPTIONS.add_argument("--headless=new")
_OPTIONS.add_argument("--no-sandbox")
_OPTIONS.add_argument("--disable-dev-sh-usage")
_OPTIONS.add_argument("--ignore-certificate-errors")
_URL_COUNT = 10
_WORKERS = 10
_TRANSFORMER = Html2TextTransformer()

_SELENIUM_WORKERS: list[webdriver.Chrome] = []


def _fetch_url(args) -> str | None:
    idx, url = args
    worker: webdriver.Chrome = _SELENIUM_WORKERS[idx % _WORKERS]
    worker.get(url)
    if worker.page_source:
        return ReadabilityDocument(worker.page_source).summary()
    return None


def google_search(query: str) -> list[dict]:
    response = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={
            "key": APP_CONFIG.google.custom_search_api_key,
            "cx": APP_CONFIG.google.custom_search_engine_id,
            "q": query,
        },
    )
    return response.json()["items"]


def internet_scrape(topic: str) -> str:
    # pylint: disable=global-statement
    global _SELENIUM_WORKERS
    _SELENIUM_WORKERS = [webdriver.Chrome(options=_OPTIONS) for _ in range(_WORKERS)]

    items = google_search(topic)
    urls = [result["link"] for result in items]
    docs = []

    with ThreadPoolExecutor(max_workers=len(_SELENIUM_WORKERS)) as executor:
        docs = executor.map(_fetch_url, enumerate(urls))

    # Transformer extracts raw text from html
    docs = _TRANSFORMER.transform_documents(
        [LangchainDocument(page_content=html) for html in docs if html]
    )

    # Drop links that reject scraper
    docs = [
        doc.page_content
        for doc in docs
        if all(
            fail_text not in doc.page_content.lower()
            for fail_text in [
                "# 403 Forbidden",
                "something went wrong",
                "sign in",
                "needs javascript to work",
                "please do not scrape our pages",
                "access_denied",
                "we use cookies",
                "on this server",
            ]
        )
    ]
    docs = docs[:_URL_COUNT]

    # Summarize to not overwhelm the model
    output = []
    for doc in docs:
        parser = PlaintextParser.from_string(doc, Tokenizer("english"))
        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, sentences_count=3)
        summary = " ".join([str(sentence) for sentence in summary])
        output.append(summary[:300] + "..." if len(summary) > 300 else summary)

    for worker in _SELENIUM_WORKERS:
        worker.quit()

    return ". ".join(output)
