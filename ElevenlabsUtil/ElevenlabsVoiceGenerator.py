import tempfile
import elevenlabs

class ElevenlabsVoiceGenerator:
    def __init__(self, api_key):
        self.api_key = api_key

    def generate_voice(self, voice_name : str, prompt: str):
        if not voice_name in ["Rachel","Domi","Bella","Antoni","Elli","Josh","Arnold","Adam","Sam"]:
            raise ValueError("Invalid voice!")

        # set_api_key(self.api_key)
        try:
            audio_bytes = elevenlabs.generate( 
              api_key = self.api_key,
              text = prompt,
              voice = voice_name,
              model = "eleven_multilingual_v1"    
            )
            
            with tempfile.NamedTemporaryFile('w', delete=False) as audio_file:
                audio_file.write(audio_bytes)            
                return audio_file
        except elevenlabs.api.error.RateLimitError as error:
            raise Exception(f"RateLimitError : {error}")