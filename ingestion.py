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
from openai import batches

from logger import (Colors,log_header,log_error, log_info, log_success, log_warning)

load_dotenv()

ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["ReQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",show_progress_bar=False, chunk_size=50, retry_min_seconds=10, retry_max_seconds=30, max_retries=5)

vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
# vectorstore = PineconeVectorStore(index_name=os.getenv('INDEX_NAME'), embedding=embeddings)  
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

async def index_documents(documents: List[Document], batch_size: int = 50):
    log_info("Indexing documents into vector database...", Colors.PURPLE)
    
    # 1. Split into batches
    batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
    
    # 2. Set concurrency limit (e.g., only 2 batches at a time)
    # and a small delay between batches to stay under Google's RPM limits
    semaphore = asyncio.Semaphore(2) 

    async def add_batch_with_throttle(batch: List[Document], batch_num: int):
        async with semaphore:
            try:
                await vectorstore.aadd_documents(batch)
                log_success(f"Batch {batch_num} indexed successfully.")
                # Small "breather" for the API
                await asyncio.sleep(2) 
                return True
            except Exception as e:
                log_error(f"Batch {batch_num} failed: {e}")
                return False

    log_info(f"Indexing {len(documents)} docs in {len(batches)} throttled batches...", Colors.PURPLE)

    # 3. Create and run tasks
    tasks = [add_batch_with_throttle(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks)

    successful_batches = sum(1 for r in results if r is True)
    log_success(f"Successfully indexed {successful_batches}/{len(batches)} batches.")

    if successful_batches == len(batches):
        log_success(f"All {len(batches)} batches indexed successfully.")    
    else:
        log_warning(f"{successful_batches} out of {len(batches)} batches indexed successfully. Check logs for details on failures.")


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
    
    #split the documents into chunks
    log_header("Document Chunking and Vector Store Ingestion Started")
    log_info("Using RecursiveCharacterTextSplitter to split documents into chunks of 1000 characters with 200 characters overlap...", Colors.PURPLE)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splitter_documents = text_splitter.split_documents(all_docs)
    log_success(f"Document chunking completed. {len(splitter_documents)} chunks created from {len(all_docs)} original documents.")
    
    await index_documents(splitter_documents, batch_size=500)

    log_header("Documentation Ingestion Process Completed")
    log_success(f"Total documents indexed: {len(splitter_documents)}. You can now query your vector database for relevant information.")
    log_info("Summary: The ingestion process successfully crawled the website, extracted content, split it into manageable chunks, and indexed it into the vector database for efficient retrieval.", Colors.GREEN)
    log_info("urls used for crawling: " + ", ".join(site_map["results"]), Colors.GREEN)
    log_info("documents extracted: " + str(len(all_docs)), Colors.GREEN)
    log_info("chunks created: " + str(len(splitter_documents)), Colors.GREEN)
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