from abc import ABC, abstractmethod
import requests



class Model(ABC):
    @abstractmethod
    def create_response(self, model_message):
        pass

    @abstractmethod
    def calculate_cost(self, input_tokens, output_tokens):
        pass



class TextGenUIAPIModel(Model):
    def create_response(self, model_message) -> dict:

        url = "http://149.36.0.216:44147/v1/chat/completions"

        headers = {"Content-Type": "application/json"}
        data = {"mode": "instruct", "messages": model_message}
        response = requests.post(url, headers=headers, json=data, verify=False)
        assistant_message = response.json()["choices"][0]["message"]["content"]

        response = {
            "content": assistant_message,
        }

        return response

    def calculate_cost(self, input_tokens, output_tokens):
        pass