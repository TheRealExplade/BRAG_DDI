import requests

from llm.interface import LLMInterface

class OllamaLLM(LLMInterface):

    def __init__(self, model="mistral"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def generate(self, prompt):

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 300,
                # Ollama defaults to a 2048-token context window when this
                # isn't set, which the prompt template's own instructions +
                # example already eat into (~500-600 tokens) before any
                # dynamic evidence is added. Set explicitly so a longer
                # mechanism report (PK reasoning can run 1000+ chars per
                # drug pair) doesn't silently get truncated by Ollama
                # itself -- which drops from context, not something we
                # control or see happen.
                "num_ctx": 4096
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