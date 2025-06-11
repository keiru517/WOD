from openai import OpenAI


class GPTHandler:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)

    def generate_response(self, prompt: str, query: str) -> str:
        response = self.client.responses.create(
            model="gpt-4.1",
            input=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": query},
            ],
        )
        return response.output_text
