import requests
import json

# Handle relative imports for direct execution
try:
    from ..config import config
except (ImportError, ValueError):
    # Fallback for direct execution
    import sys
    from pathlib import Path
    # Add the backend directory to the path
    backend_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(backend_dir))
    from config import config

class CohereClient:
    def __init__(self):
        # Check if required config is available before initializing
        if not config.COHERE_API_KEY:
            print("Warning: Cohere API key not found. Cohere client will be in offline mode.")
            self.api_key = None
            return

        self.api_key = config.COHERE_API_KEY
        self.base_url = "https://api.cohere.ai/v1"
        self.embedding_model = "embed-english-v3.0"
        self.generation_model = "command-r-plus-08-2024"
        self.fallback_generation_model = "command-r"

    def embed_query(self, query):
        """
        Generate embedding for a single query using Cohere's API via HTTP requests
        """
        if not self.api_key:
            print("Cohere API key not available, returning dummy embedding")
            return [0.0] * 1024  # Return a dummy embedding

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "texts": [query],
                "model": self.embedding_model,
                "input_type": "search_query"
            }

            response = requests.post(f"{self.base_url}/embed", headers=headers, json=data)

            if response.status_code == 200:
                result = response.json()
                return result["embeddings"][0]  # Return the first embedding
            else:
                print(f"Cohere embedding request failed: {response.status_code} - {response.text}")
                return [0.0] * 1024  # Return dummy embedding on failure
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 1024  # Return dummy embedding on exception

    def generate(self, prompt, max_tokens=500, temperature=0.3):
        """
        Generate text using Cohere's API via HTTP requests
        """
        if not self.api_key:
            print("Cohere API key not available, returning default response")
            return "Cohere service is temporarily unavailable. Please contact the administrator."

        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            data = {
                "model": self.generation_model,
                "message": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            }

            response = requests.post(f"{self.base_url}/chat", headers=headers, json=data)

            if response.status_code == 200:
                result = response.json()
                return result.get("text", result.get("response", "No response generated"))
            else:
                # Try with fallback model
                data["model"] = self.fallback_generation_model
                response = requests.post(f"{self.base_url}/chat", headers=headers, json=data)

                if response.status_code == 200:
                    result = response.json()
                    return result.get("text", result.get("response", "No response generated"))
                else:
                    print(f"Cohere generation request failed: {response.status_code} - {response.text}")
                    return "Unable to generate a response at this time."
        except Exception as e:
            print(f"Error during generation: {e}")
            return "Unable to generate a response at this time."

    def embed(self, texts, input_type="search_document"):
        """
        Generate embeddings for the given texts using Cohere's embed model
        """
        if self.client is None:
            print("Cohere client not available, returning dummy embeddings")
            # Return dummy embeddings (same dimension as real embeddings)
            return [[0.0] * 1024 for _ in range(len(texts))]

        try:
            response = self.client.embed(
                texts=texts,
                model=self.embedding_model,
                input_type=input_type
            )
            return response.embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return dummy embeddings as fallback
            return [[0.0] * 1024 for _ in range(len(texts))]

    def embed_query(self, query):
        """
        Generate embedding for a single query
        """
        return self.embed([query], input_type="search_query")[0]

    def generate(self, prompt, max_tokens=500, temperature=0.3):
        """
        Generate text using Cohere's command-r model (updated to use Chat API)
        """
        if self.client is None:
            print("Cohere client not available, returning default response")
            return "Cohere service is temporarily unavailable. Please contact the administrator."

        try:
            # Use the new Chat API instead of the deprecated Generate API
            response = self.client.chat(
                model=self.generation_model,
                message=prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            if response.text:
                return response.text
            else:
                # Fallback to command-r-plus if command-r fails
                fallback_response = self.client.chat(
                    model=self.fallback_generation_model,
                    message=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )

                if fallback_response.text:
                    return fallback_response.text
                else:
                    return "Unable to generate a response at this time."

        except Exception as e:
            print(f"Error during generation: {e}")
            # Fallback to command-r-plus if command-r fails
            try:
                if self.client is not None:
                    fallback_response = self.client.chat(
                        model=self.fallback_generation_model,
                        message=prompt,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )

                    if fallback_response and fallback_response.text:
                        return fallback_response.text
                    else:
                        return "Unable to generate a response at this time."
                else:
                    return "Unable to generate a response at this time due to service unavailability."
            except Exception as fallback_error:
                print(f"Fallback generation also failed: {fallback_error}")
                return "Unable to generate a response at this time."

# Create a singleton instance
cohere_client = CohereClient()