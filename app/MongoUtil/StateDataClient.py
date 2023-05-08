from pymongo import MongoClient

class StateDataClient:
    def __init__(self, connection_string : str, database : str, collection="PromptResponseState"):
        self.client = MongoClient(connection_string)
        self.promptstate = self.client[database][collection]

    def save_prompt_response(self, prompt: str, keyword: str, response: str, prompttype: str):
        try:
            self.promptstate.insert_one({
                'prompt': prompt,
                'keyword': keyword,
                'prompttype': prompttype,
                'response': response
            })
        except Exception as err:
            print(f"StateDataClient save_prompt_response error {err}")
            print(err)

    def read_description_from_prompt(self, keyword: str):
        try:
            qry = {'keyword': {'$regex': keyword.strip()}}
            promptresponsestate = self.promptstate.find_one(qry)
            if promptresponsestate:
                return promptresponsestate["prompt"], promptresponsestate["response"]
            else:
                return "", ""
        except Exception as err:
            print(f"StateDataClient read_description_from_prompt error {err}")
            return "", ""

    def list_saved_prompts(self, prompttype: str):
        try:
            prompt_examples = []

            #promptlist = self.promptstate.find({})            
            qry = {'prompttype': {'$regex': prompttype.strip()}}
            promptlist = self.promptstate.find(qry)
            for prompt in promptlist:
                prompt_examples.append([prompt['keyword']])
            return prompt_examples
        except Exception as err:
            print(f"StateDataClient list_saved_prompts error {err}")
            return []