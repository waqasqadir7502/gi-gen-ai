import cohere

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
        self.client = cohere.Client(config.COHERE_API_KEY)
        self.embedding_model = "embed-english-v3.0"
        self.generation_model = "command-r-08-2024"  # Updated to available model
        self.fallback_generation_model = "command-light"  # Updated to available fallback

    def embed(self, texts, input_type="search_document"):
        """
        Generate embeddings for the given texts using Cohere's embed model
        """
        try:
            response = self.client.embed(
                texts=texts,
                model=self.embedding_model,
                input_type=input_type
            )
            return response.embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise

    def embed_query(self, query):
        """
        Generate embedding for a single query
        """
        return self.embed([query], input_type="search_query")[0]

    def generate(self, prompt, max_tokens=500, temperature=0.3):
        """
        Generate text using Cohere's command-r model (updated to use Chat API)
        """
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
            except Exception as fallback_error:
                print(f"Fallback generation also failed: {fallback_error}")
                return "Unable to generate a response at this time."

# Create a singleton instance
cohere_client = CohereClient()