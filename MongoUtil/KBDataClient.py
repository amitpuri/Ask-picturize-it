from pymongo import MongoClient

class KBDataClient:
    def __init__(self, connection_string : str, database : str, collection="KBSearchData"):
        self.client = MongoClient(connection_string)
        self.KBSearchData = self.client[database][collection]

    def save_kb_searchdata(self, output):
        try:
            self.KBSearchData.insert_many(output)
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