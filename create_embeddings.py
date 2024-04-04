import argparse
import pickle
import requests
import xmltodict

from dotenv import load_dotenv
from bs4 import BeautifulSoup
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()

def extract_text_from(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()

    lines = (line.strip() for line in text.splitlines())
    return '\n'.join(line for line in lines if line)


def extract_urls_from_sitemap(sitemap_url):
    r = requests.get(sitemap_url)
    xml = r.text
    raw = xmltodict.parse(xml)
    urls = []
    if 'urlset' in raw and 'url' in raw['urlset']:
        url_infos = raw['urlset']['url']
        url_infos = url_infos if isinstance(url_infos, list) else [url_infos]
        urls = [info['loc'] for info in url_infos]
    elif 'sitemapindex' in raw and 'sitemap' in raw['sitemapindex']:  # this is a common alternative structure
        url_infos = raw['sitemapindex']['sitemap']
        url_infos = url_infos if isinstance(url_infos, list) else [url_infos]
        urls = [info['loc'] for info in url_infos]
    return urls



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedding website content')
    parser.add_argument('-s', '--sitemap', type=str, required=False,
            help='URL zur Sitemap.xml Datei', default='https://www.fundraisingscript.com/sitemap.xml')
    parser.add_argument('-f', '--filter', type=str, required=False,
            help='Dieser Text muss in der URL enthalten sein, damit die URL berücksichtigt wird',
            default='https://www.neuharlingersiel.de/veranstaltung')
    args = parser.parse_args()

    pages = []
    sitemap_urls = extract_urls_from_sitemap(args.sitemap)
    for sitemap_url in sitemap_urls:
        if sitemap_url.endswith('sitemap.xml'):
            urls = extract_urls_from_sitemap(sitemap_url)
            for url in urls:
                if args.filter in url:
                    print(f"URL passed filter: {url}")  # print URLs that pass the filter
                    pages.append({'text': extract_text_from(url), 'source': url})
        else:
            if args.filter in sitemap_url:
                print(f"URL passed filter: {sitemap_url}")  # print URLs that pass the filter
                pages.append({'text': extract_text_from(sitemap_url), 'source': sitemap_url})

    text_splitter = CharacterTextSplitter(chunk_size=1500, separator="\n")
    docs, metadatas = [], []
    for page in pages:
        splits = text_splitter.split_text(page['text'])
        docs.extend(splits)
        metadatas.extend([{"source": page['source']}] * len(splits))
        print(f"Split {page['source']} into {len(splits)} chunks")

    print(f"Length of docs: {len(docs)}")  # print length of docs

    store = FAISS.from_texts(docs, OpenAIEmbeddings(), metadatas=metadatas)
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)