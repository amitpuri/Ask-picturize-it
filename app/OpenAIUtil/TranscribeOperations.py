from OpenAIUtil.Operations import *

import openai

class TranscribeOperations(Operations):
    def __init__(self, api_key: str, org_id: str):
        if org_id is not None:
            openai.organization = org_id
        openai.api_key = api_key
    
    def transcribe(self, audio_file: str):
        try: 
            if audio_file is not None and openai.api_key is not None:
                audio = open(audio_file, "rb")
                transcript = openai.Audio.transcribe("whisper-1", audio)
                return transcript["text"], transcript["text"]
            else:
                return "", ""
        except openai.error.OpenAIError as error_except:
            print("TranscribeOperations transcribe")
            print(error_except.http_status)
            print(error_except.error)
            return error_except.error["message"], ""

