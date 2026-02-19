import asyncio
import ssl
import os
from typing import List, Dict, Any 
import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  
from langchain_core.documents import Document
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyMap, TavilyExtract

from logger import (Colors,log_header,log_error, log_info, log_success, log_warning)

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["ReQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-3-small",show_progress_bar=False, chunk_size=50, retry_min_seconds=10, retry_max_seconds=30, max_retries=5)

vectorstore = PineconeVectorStore(index_name="langchain-doc-index", embedding=embeddings)  
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20,max_pages=1000)
tavily_crawl = TavilyCrawl()

def chunk_urls(urls: List[str], chunk_size: int = 20) -> List[List[str]]:
    """Split URLs into chunks of specified size."""
    chunks = []
    for i in range(0, len(urls), chunk_size):
        chunk = urls[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

async def extract_batch(urls: List[str], batch_num: int) -> List[Dict[str, Any]]:
    """Extract documents from a batch of URLs."""
    try:
        print(f"🔄 Processing batch {batch_num} with {len(urls)} URLs")
        docs = await tavily_extract.ainvoke(input={"urls": urls})
        results = docs.get('results', [])
        print(f"✅ Batch {batch_num} completed - extracted {len(results)} documents")
        return results
    except Exception as e:
        print(f"❌ Batch {batch_num} failed: {e}")
        return []
    
async def async_extraction(url_chunks: List[List[str]]) -> List[Dict[str, Any]]:
    log_info("Document extraction phase started with TavilyExtract...", Colors.PURPLE)

    log_info(f"Processing {len(url_chunks)} batches of URLs asynchronously...", Colors.PURPLE)

    tasks = [extract_batch(urls, idx + 1) for idx, urls in enumerate(url_chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out any exceptions and flatten the results
    extracted_docs = []
    failed_batches = 0
    for result in results:
        if isinstance(result, Exception):
            failed_batches += 1
            log_error(f"Batch failed with error: {result}")
        else:
           for extracted in result:
                document = Document(
                    page_content=extracted.get("raw_content", ""),
                    metadata={"url": extracted.get("url", "")})
                extracted_docs.append(document)

    log_success(f"Document extraction completed. {len(extracted_docs)} documents extracted with {failed_batches} failed batches.")
    if failed_batches > 0:
        log_warning(f"{failed_batches} batches failed during extraction. Consider reviewing the errors for troubleshooting.")
    return extracted_docs


async def main():
    """Main async function to orchestrate the ingestion process."""
    log_header("Documentation Ingestion Process Started")
    
    log_info("TavilyMap: Starting web crawling with Tavily from https://python.langchain.com..."
             , Colors.PURPLE)
    
    site_map = tavily_map.invoke("https://python.langchain.com/")
    log_success(f"TavilyMap: Crawling completed. {len(site_map['results'])} URLs found.")


    # Split url into batches of 20
    url_chunks = chunk_urls(list(site_map["results"]), chunk_size=20)
    log_info(f"Processing split {len(site_map['results'])} urls into  {len(url_chunks)} batches of URLs with TavilyExtract...", Colors.PURPLE)


    all_docs = await async_extraction(url_chunks)
    log_info(f"Total documents extracted: {len(all_docs)}. Proceeding to store in vector database...", Colors.PURPLE)
    # Crwal the website and get the list of URLs

    # res = tavily_crawl.invoke({
    #     "url":"https://python.langchain.com/",
    #     "max_depth":1,
    #     "instructions":"content on ai agents",
    #     "extract_depth":"advanced",
    # }
    # )

    # allowed_urls = [Document(page_content=results["raw_content"], metadata={"url": results["url"]}) for results in res["results"]]
    # log_success(f"Crawling completed. {len(allowed_urls)} URLs found.")


if __name__ == "__main__":
    asyncio.run(main())