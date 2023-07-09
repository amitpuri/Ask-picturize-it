import assemblyai as aai

class AssemblyAITranscriber:
    def __init__(self, api_key):
        self.api_key = api_key

    def transcribe(self, audio_file: str):
        if self.api_key and audio_file:
            aai.settings.api_key = self.api_key
            transcriber = aai.Transcriber()
            
            config = aai.TranscriptionConfig(
                punctuate = True,
                format_text = True
            )
            transcript = transcriber.transcribe(audio_file, config)
                        
            return transcript.text