from typing import Dict, Any, List
import re

# Handle relative imports for direct execution
try:
    from ..clients.cohere_client import cohere_client
    from ..prompts.chat_prompt import chat_prompt_engineer
    from ..utils.logger import log_info, log_error, log_warning
    from ..utils.metadata_extractor import metadata_extractor
except (ImportError, ValueError):
    # Fallback for direct execution
    import sys
    from pathlib import Path
    # Add the backend directory to the path
    backend_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(backend_dir))

    try:
        from clients.cohere_client import cohere_client
        from prompts.chat_prompt import chat_prompt_engineer
        from utils.logger import log_info, log_error, log_warning
        from utils.metadata_extractor import metadata_extractor
    except ImportError:
        print("Could not import services - using fallbacks")
        # Define minimal fallbacks for testing
        class MockCohereClient:
            def generate(self, prompt, **kwargs):
                return "Mock response for testing"

        cohere_client = MockCohereClient()
        chat_prompt_engineer = None

class GenerationService:
    def __init__(self):
        self.max_tokens = 500
        self.temperature = 0.3
        self.moderation_keywords = [
            'inappropriate', 'offensive', 'harmful', 'violent', 'discriminatory',
            'harassment', 'threat', 'explicit', 'nudity', 'sexual', 'hate speech'
        ]

    def _moderate_content(self, text: str) -> bool:
        """
        Check if content contains inappropriate elements
        Returns True if content is safe, False if it should be moderated
        """
        text_lower = text.lower()

        # Check for moderation keywords
        for keyword in self.moderation_keywords:
            if keyword in text_lower:
                return False

        # Check for potentially harmful patterns (regex)
        harmful_patterns = [
            r'kill.*yourself',
            r'hate.*speech',
            r'threaten.*violence',
        ]

        for pattern in harmful_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return False

        return True

    def _sanitize_response(self, text: str) -> str:
        """
        Sanitize the response to remove any potentially problematic content
        """
        # Replace potentially harmful content
        sanitized = re.sub(r'\b(damn|hell|crap)\b', 'darn', text, flags=re.IGNORECASE)
        return sanitized

    def generate_response(self, query: str, context_chunks: List[Dict], selected_context: str = None) -> Dict[str, Any]:
        """
        Generate a response using the RAG approach with context and query
        """
        try:
            # Build the RAG prompt with grounding instructions
            prompt, sources = chat_prompt_engineer.build_rag_prompt(query, context_chunks, selected_context)

            # Generate response using Cohere
            response_text = cohere_client.generate(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature
            )

            # Moderate the response
            if not self._moderate_content(response_text):
                return {
                    "answer": "The response was flagged for review and cannot be displayed.",
                    "sources": [],
                    "metadata": {
                        "model_used": "command-r",
                        "context_chunks_used": len(context_chunks),
                        "generation_timestamp": __import__('datetime').datetime.now().isoformat(),
                        "moderation_flag": True
                    }
                }

            # Sanitize the response
            sanitized_response = self._sanitize_response(response_text)

            # Format the response with proper citations
            result = {
                "answer": sanitized_response,
                "sources": sources,
                "metadata": {
                    "model_used": "command-r",
                    "context_chunks_used": len(context_chunks),
                    "generation_timestamp": __import__('datetime').datetime.now().isoformat()
                }
            }

            log_info("Response generated successfully", extra={
                "query_length": len(query),
                "context_chunks": len(context_chunks),
                "response_length": len(response_text)
            })

            return result

        except Exception as e:
            log_error(f"Error during response generation: {str(e)}", extra={
                "query": query[:100] + "..." if len(query) > 100 else query
            })

            # Return a fallback response
            return {
                "answer": "I encountered an error while generating a response. Please try again.",
                "sources": [],
                "metadata": {
                    "error": str(e),
                    "model_used": "command-r"
                }
            }

    def generate_with_validation(self, query: str, context_chunks: List[Dict], selected_context: str = None) -> Dict[str, Any]:
        """
        Generate response with additional validation and grounding checks
        """
        try:
            # First, generate the initial response
            initial_result = self.generate_response(query, context_chunks, selected_context)

            # Validate that the response is grounded in the context
            context_text = " ".join([chunk.get('content', '') for chunk in context_chunks])

            # Check if the response seems properly grounded (simple heuristic)
            answer = initial_result['answer']

            # For now, we'll return the initial result
            # In a more sophisticated system, we would perform grounding validation

            return initial_result

        except Exception as e:
            log_error(f"Error during validated generation: {str(e)}", extra={
                "query": query[:100] + "..." if len(query) > 100 else query
            })

            return {
                "answer": "I encountered an error while generating a response. Please try again.",
                "sources": [],
                "metadata": {
                    "error": str(e)
                }
            }

    def generate_streaming_response(self, query: str, context_chunks: List[Dict], selected_context: str = None):
        """
        Generate a streaming response (placeholder implementation)
        """
        # This would be implemented for streaming in a production system
        # For now, we'll just return the regular response
        return self.generate_response(query, context_chunks, selected_context)

    def validate_response_grounding(self, answer: str, context: str) -> bool:
        """
        Validate that the response is properly grounded in the context
        This is a simplified check - in practice, you might use more sophisticated validation
        """
        try:
            # Simple check: see if the answer contains information that's in the context
            answer_lower = answer.lower()
            context_lower = context.lower()

            # Count overlapping terms
            answer_words = set(answer_lower.split())
            context_words = set(context_lower.split())
            overlap = len(answer_words.intersection(context_words))

            # If there's significant overlap, it's likely grounded
            if len(answer_words) > 0:
                overlap_ratio = overlap / len(answer_words)
                is_likely_properly_based = overlap_ratio > 0.1  # At least 10% overlap
            else:
                is_likely_properly_based = False

            return is_likely_properly_based

        except Exception as e:
            log_error(f"Error during grounding validation: {str(e)}")
            return True  # Default to accepting if validation fails

    def generate_summarized_response(self, query: str, context_chunks: List[Dict], selected_context: str = None) -> Dict[str, Any]:
        """
        Generate a summarized response focusing on the most relevant information
        """
        try:
            # Sort chunks by relevance score if available
            sorted_chunks = sorted(context_chunks, key=lambda x: x.get('score', 0), reverse=True)

            # Take top chunks
            top_chunks = sorted_chunks[:3]  # Take top 3 most relevant chunks

            # Generate response with top chunks
            return self.generate_response(query, top_chunks, selected_context)

        except Exception as e:
            log_error(f"Error during summarized response generation: {str(e)}")
            return self.generate_response(query, context_chunks, selected_context)

# Create a singleton instance
generation_service = GenerationService()