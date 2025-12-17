from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker 
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.schema import Document
from bs4 import BeautifulSoup
from trafilatura import fetch_url, extract,bare_extraction
from dotenv import load_dotenv
from pathlib import Path
import subprocess
import json
import os
import re



path = Path(__file__).parent.parent.parent

dotenv_path = path / '.env'

load_dotenv(dotenv_path)

openai_api = os.getenv("OPENAI_API")
headers = os.getenv("USER-AGENT")


class Embedder:
    def __init__(self):
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )

        self.vectordb = Chroma(
            persist_directory=Path("~/embedding_pipeline/chroma").expanduser(),
            embedding_function=self.embeddings_model
        )

        self.in_mem_vectordb = InMemoryVectorStore(
            embedding = self.embeddings_model
        )

        self.retriever = self.vectordb.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 1})
        
    
        
    def embed_short_term(self, chat_history_as_str):

        hard_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,     
            chunk_overlap=15
        )

        semantic_splitter = SemanticChunker(
            self.embeddings_model, breakpoint_threshold_type="percentile"
        )

        final_splitter = RecursiveCharacterTextSplitter(
            chunk_size=50,
            chunk_overlap=6
        )
        document = Document(page_content=chat_history_as_str)

        raw_documents = hard_splitter.split_documents(document)
        semantic_documents = semantic_splitter.split_documents(raw_documents)
        final_documents = final_splitter.split_documents(semantic_documents)

        print("=========================")
        print(final_documents)
        print("=========================")

        self.in_mem_vectordb.add_documents(document=[final_documents])





    def load_site_into_db(self):
        
        hard_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,     # hard ceiling
            chunk_overlap=150
        )

        semantic_splitter = SemanticChunker(
            self.embeddings_model, breakpoint_threshold_type="percentile"
        )

        final_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        clean_sites = []

        while True: 
            url = input("Enter url: ")
            if url == "EXIT":
                break
            if not url.startswith("https://"):
                print("Invalid or unsafe link. Please try again.")
                continue
            retrieved_html = fetch_url(url)
            clean_text = bare_extraction(retrieved_html,
                                include_comments=False,
                                include_tables=True,
                                no_fallback=False,
                                with_metadata=True)
            
            site_body = clean_text.text
            site_title = clean_text.title
            site_url = clean_text.url


            clean_doc = Document(page_content=site_body, 
                                 metadata={"source": f"{site_url}", 
                                            "title": f"{site_title}",
                                            "tags": f"{clean_text.tags}",
                                            "description": f"{clean_text.description}"})           
            clean_sites.append(clean_doc)
            print(clean_doc)

        raw_documents = hard_splitter.split_documents(clean_sites)
        semantic_documents = semantic_splitter.split_documents(raw_documents)
        final_documents = final_splitter.split_documents(semantic_documents)

        db = Chroma.from_documents(
            documents=final_documents,
            embedding=self.embeddings_model,
            persist_directory=Path("~/embedding_pipeline/chroma").expanduser()
        )

