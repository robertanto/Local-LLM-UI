import os
import random
import requests
import gradio as gr

base_url = os.environ['INFERENCE_API_ENDPOINT'] + '/chat'

def chatbot_response(message, history):
    # Call API to get the answer
    response = requests.post(base_url, json={'message': message, 'history': history})
    answer = response.json()['answer']

    return answer

demo = gr.ChatInterface(chatbot_response)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)