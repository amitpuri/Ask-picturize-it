import os
import json
import arxiv
from Utils.AskPicturizeIt import *
from youtube_search import YoutubeSearch
from Utils.YouTubeSummarizer import *
from Utils.PDFSummarizer import *
from Utils.RapidapiUtil import *
from MongoUtil.KBDataClient import *

class KnowledgeBase:
    def __init__(self):
        self.ask_picturize_it = AskPicturizeIt()
    
    def get_private_mongo_config(self):
        return os.getenv("P_MONGODB_URI"), os.getenv("P_MONGODB_DATABASE")

    def get_searchData_by_uri(self, uri: str):
        try:
            connection_string, database = self.get_private_mongo_config()           
            kb_data_client = KBDataClient(connection_string, database)
            title, summary = kb_data_client.search_data_by_uri(uri)        
        except Exception as exception:
            print(f"Exception Name: {type(exception).__name__}")
            print(exception)
            title = ""
            summary = ""
            pass
        finally:
            return title, summary
    
    def extract_youtube_attributes(self, keyword, output):
        videos = []
        for video in output.videos:
            print(video)
            video_id = video["id"]
            url = f"https://www.youtube.com/watch?v={video_id}"
            videos.append({
                'kbtype': "youtube",
                'keyword': keyword ,
                'title': video["title"],
                'url': url,
                'summary': None
            })
        return videos
    
    def extract_arxiv_attributes(self, keyword, output):
        papers = []
        for pdf in output.results():
            papers.append({
                'kbtype': "pdf",
                'keyword': keyword ,
                'title': pdf.title,
                'url': pdf.pdf_url,
                'summary': pdf.summary
            })
        return papers
    
    def pdf_search_data_by_uri(self, uri: str):
        title, summary = self.get_searchData_by_uri(uri)
        return uri, title, summary
        
    def youtube_search_data_by_uri(self, uri: str):
        title, summary = self.get_searchData_by_uri(uri)
        return uri, title
    
    
    def kb_search(self, keyword: str, select_medium, max_results: int):
        connection_string, database = self.get_private_mongo_config()
        kb_data_client = KBDataClient(connection_string, database)
        if select_medium == 0:
            output = YoutubeSearch(keyword, max_results=max_results)
            try:
                videos = self.extract_youtube_attributes(keyword, output)
                kb_data_client.save_kb_searchdata(videos)
            except Exception as exception:
                print(f"Exception Name: {type(exception).__name__}")
                print(exception)
                pass
    
            return output.to_json()
        elif select_medium == 1:
            output = arxiv.Search(
              query = keyword,
              max_results = max_results,
              sort_by = arxiv.SortCriterion.SubmittedDate
            )
            try:
                papers = self.extract_arxiv_attributes(keyword, output)
                kb_data_client.save_kb_searchdata(papers)
            except Exception as exception:
                print(f"Exception Name: {type(exception).__name__}")
                print(exception)
                pass
            return output.results()
        elif select_medium == 2:
            outputs = []
            for page_title in wikipedia.search(keyword, results=max_results):
                try:
                    page = wikipedia.page(page_title)            
                    outputs.append(page.url)            
                except (wikipedia.exceptions.PageError, wikipedia.exceptions.DisambiguationError):
                    pass
            return outputs
    
    def youtube_summarizer_handler(self, api_key, url):    
        if api_key:
            if url and len(url)>0:
                youtube_summarizer = YouTubeSummarizer()
                youtube_summarizer.setOpenAIConfig(api_key)
                return self.ask_picturize_it.TRANSCRIBE_OUTPUT_INFO,  youtube_summarizer.summarize(url)
            else:
                return "No URL",  ""
        else:
            return self.ask_picturize_it.NO_API_KEY_ERROR, ""
    
    def youtube_transcribe_handler(self, api_key, url):    
        if api_key:
            if url and len(url)>0:
                youtube_summarizer = YouTubeSummarizer()
                youtube_summarizer.setOpenAIConfig(api_key)
                return self.ask_picturize_it.TRANSCRIBE_OUTPUT_INFO,  youtube_summarizer.transcribe(url)
            else:
                return "No URL",  ""
        else:
            return self.ask_picturize_it.NO_API_KEY_ERROR, ""
    
    def pdf_summarizer_handler(self, api_key, url):    
        if api_key:
            if url and len(url)>0:
                pdf_summarizer = PDFSummarizer()
                pdf_summarizer.setOpenAIConfig(api_key)
                #TO DO
                return self.ask_picturize_it.PDF_OUTPUT_INFO, pdf_summarizer.summarize(url)
            else:
                return "No URL",  ""
        else:
            return self.ask_picturize_it.NO_API_KEY_ERROR, ""
    
    def article_summarize_handler(self, rapidapi_api_key, article_link, length):
        rapidapi_util = RapidapiUtil()
        if rapidapi_api_key:
            if article_link and len(article_link)>0:
                response = rapidapi_util.article_rapidapi_api("summarize", rapidapi_api_key, article_link, "summary", length)
                return response, ""
            else:            
                return "No URL",  ""
        else:
            return "", self.ask_picturize_it.NO_RAPIDAPI_KEY_ERROR 
            
    def article_extract_handler(self, rapidapi_api_key, article_link):
        rapidapi_util = RapidapiUtil()
        if rapidapi_api_key:
            if article_link and len(article_link)>0:
                response = rapidapi_util.article_rapidapi_api("extract", rapidapi_api_key, article_link, "content")
                return response, ""
            else:
                return "No URL",  ""
        else:
            return "", self.ask_picturize_it.NO_RAPIDAPI_KEY_ERROR
            
    def PDF_Examples(self):
        connection_string, database = self.get_private_mongo_config()           
        kb_data_client = KBDataClient(connection_string, database)
        return kb_data_client.list_kb_searchData("pdf")
    
    def YouTube_Examples(self):
        connection_string, database = self.get_private_mongo_config()           
        kb_data_client = KBDataClient(connection_string, database)
        return kb_data_client.list_kb_searchData("youtube")