from abc import ABC

class Operations(ABC):
    def usage_metric(response: str):
        prompt_tokens = response["usage"]["prompt_tokens"]
        completion_tokens = response["usage"]["completion_tokens"]
        total_tokens = response["usage"]["total_tokens"]
        return prompt_tokens, completion_tokens, total_tokens