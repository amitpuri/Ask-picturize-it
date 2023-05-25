from langchain.document_loaders import YoutubeLoader
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter

class YouTubeSummarizer:
    def setOpenAIConfig(self, OPENAI_API_KEY):
        self.openai_api_key = OPENAI_API_KEY
        
    def transcribe(self, url):
        llm = OpenAI(temperature=0, openai_api_key=self.openai_api_key)
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        result = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_documents(result)
        return texts

    def summarize(self, url):
        llm = OpenAI(temperature=0, openai_api_key=self.openai_api_key)
        loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
        result = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_documents(result)
        chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
        return chain.run(texts[:4])
