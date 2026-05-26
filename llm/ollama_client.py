import requests

class OllamaLLM:

    def __init__(self, model="phi3"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 300
            }
        }

        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=120
            )

            data = response.json()

            if "response" not in data:
                print("LLM ERROR:", data)

                return """
Explanation: Unable to generate response
Mechanism: Unknown
Risk Level: UNKNOWN
Recommendation: Manual review required
Alternatives: None
Confidence: LOW
Confidence Reason: Model generation failed
Reasoning: Ollama runtime error
"""

            return data["response"]

        except Exception as e:

            print("OLLAMA EXCEPTION:", e)

            return """
Explanation: Unable to generate response
Mechanism: Unknown
Risk Level: UNKNOWN
Recommendation: Manual review required
Alternatives: None
Confidence: LOW
Confidence Reason: Runtime exception
Reasoning: LLM connection failed
"""