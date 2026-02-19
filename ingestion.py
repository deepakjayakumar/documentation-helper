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


async def main():
    """Main async function to orchestrate the ingestion process."""
    log_header("Documentation Ingestion Process Started")
    
    log_info("TavilyMap: Starting web crawling with Tavily from https://python.langchain.com..."
             , Colors.PURPLE)

    # Crwal the website and get the list of URLs

    res = tavily_crawl.invoke({
        "url":"https://python.langchain.com/",
        "max_depth":1,
        "instructions":"content on ai agents",
        "extract_depth":"advanced",
    }
    )

    allowed_urls = [Document(page_content=results["raw_content"], metadata={"url": results["url"]}) for results in res["results"]]
    log_success(f"Crawling completed. {len(allowed_urls)} URLs found.")


if __name__ == "__main__":
    asyncio.run(main())