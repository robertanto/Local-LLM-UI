import os 
import requests

class HuggingFaceInferenceAPI:

    def __init__(self, 
                 model_id, 
                 model_args={"max_new_tokens":100,"temperature":0.7}, 
                 api_token=None) -> None:
        if api_token is None:
            self.api_token = os.environ['HF_INFERENCE_API_TOKEN']
        else:
            self.api_token = api_token

        self.model_id = model_id
        self.model_args = model_args
        
    def query(self, prompt):
        payload = {"inputs":prompt,"parameters":self.model_args}
        headers = {"Authorization": f"Bearer {self.api_token}"}
        API_URL = f"https://api-inference.huggingface.co/models/{self.model_id}"
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()[0]['generated_text'][len(prompt):]