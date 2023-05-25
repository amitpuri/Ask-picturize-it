# credit https://github.com/gkamradt/langchain-tutorials

from langchain.document_loaders import OnlinePDFLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.summarize import load_summarize_chain

import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

class PDFSummarizer:
    def setOpenAIConfig(self, OPENAI_API_KEY):
        self.openai_api_key = OPENAI_API_KEY
        
    def summarize(self, url):
        loader = OnlinePDFLoader(url)
        pages = loader.load()
        text = ""
        for page in pages:
            text += page.page_content
    
        text = text.replace('\t', ' ')
        text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", "\t"], chunk_size=10000, chunk_overlap=3000)
        docs = text_splitter.create_documents([text])
        output = "TO DO"
        return docs