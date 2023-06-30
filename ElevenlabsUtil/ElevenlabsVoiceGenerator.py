import tempfile
import elevenlabs
import os
from elevenlabs import voices, set_api_key

class ElevenlabsVoiceGenerator:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_voice(self, voice_name : str, prompt: str):
        if not voice_name in ["Rachel","Domi","Bella","Antoni","Elli","Josh","Arnold","Adam","Sam"]:
            raise ValueError("Invalid voice!")

        # set_api_key(self.api_key)
        audio_bytes = None
        try:
            audio_bytes = elevenlabs.generate( 
              api_key = self.api_key,
              text = prompt,
              voice = voice_name,
              model = "eleven_multilingual_v1"    
            )
        except elevenlabs.api.error.RateLimitError as error:
            raise Exception(f"RateLimitError : {error}")

        if audio_bytes:
            filename = tempfile.NamedTemporaryFile('w', delete=False).name             
            with open(filename, "wb") as audio_file:
                audio_file.write(audio_bytes)            
                return filename
        
    def voices_list(self):        
        #set_api_key(os.getenv("ELEVEN_API_KEY"))
        set_api_key(self.api_key)
        return [voice.name for voice
 in voices()]