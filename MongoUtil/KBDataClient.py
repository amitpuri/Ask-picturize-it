from pymongo import MongoClient

class KBDataClient:
    def __init__(self, connection_string : str, database : str, collection="KBSearchData"):
        self.client = MongoClient(connection_string)
        self.KBSearchData = self.client[database][collection]

    def search_data_by_uri(self, uri):
        try:            
            qry = {'url': uri.strip()}
            KBSearchData = self.KBSearchData.find_one(qry)
            if KBSearchData:
                return KBSearchData['title'], KBSearchData['summary']
        except Exception as err:
            print(f"KBDataClient search_data_by_uri error {err}")
            return "",""
        
    def save_kb_searchdata(self, output):
        try:
            if output or len(output)>0:
                self.KBSearchData.insert_many(output)
            else:
                print(f"KBDataClient save_kb_searchdata error None or non-empty list")
        except Exception as err:
            print(f"KBDataClient save_kb_searchdata error {err}")
            print(err)

    def list_kb_searchData(self, kbtype: str):
        try:
            kb_searchData_examples = []
            qry = {'kbtype': {'$regex': kbtype.strip()}}
            KBSearchDatalist = self.KBSearchData.find(qry)
            for KBSearchData in KBSearchDatalist:
                if not KBSearchData['url'] in kb_searchData_examples:
                    kb_searchData_examples.append([KBSearchData['url']])

            return kb_searchData_examples
        except Exception as err:
            print(f"KBDataClient list_kb_searchData error {err}")
            return []