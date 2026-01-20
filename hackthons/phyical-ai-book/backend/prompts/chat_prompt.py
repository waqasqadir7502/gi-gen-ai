from typing import List, Dict, Any

# Handle relative import for direct execution
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

class ChatPromptEngineer:
    def __init__(self):
        self.max_context_length = 2000  # Maximum context length in characters
        self.citation_format = "[SOURCE: {source}]"

    def build_rag_prompt(self, query: str, context_chunks: List[Dict], selected_context: str = None) -> str:
        """
        Build a RAG prompt with strict grounding instructions
        """
        # Combine context chunks into a single context string
        context_texts = []
        sources = []

        for chunk in context_chunks:
            content = chunk.get('content', '')
            if content.strip():
                context_texts.append(content)

                # Extract source information
                source_info = chunk.get('metadata', {})
                source_path = source_info.get('relative_path', source_info.get('file_path', 'Unknown'))
                source_heading = source_info.get('source_heading', '')

                if source_heading:
                    source_ref = f"{source_path}#{source_heading}"
                else:
                    source_ref = source_path

                sources.append(source_ref)

        combined_context = "\n\n".join(context_texts)

        # Add selected context if provided
        if selected_context:
            combined_context = f"Additional Context:\n{selected_context}\n\nMain Context:\n{combined_context}"

        # Ensure context doesn't exceed maximum length
        if len(combined_context) > self.max_context_length:
            combined_context = combined_context[:self.max_context_length]
            # Try to cut at a sentence boundary
            last_period = combined_context.rfind('.')
            if last_period > 0:
                combined_context = combined_context[:last_period + 1]

        # Build the prompt with grounding instructions
        prompt = f"""You are a helpful AI assistant for the Physical AI Book. Answer the user's question based strictly on the provided context.

CONTEXT:
{combined_context}

INSTRUCTIONS:
- Answer the question based ONLY on the provided context
- If the context doesn't contain information to answer the question, say "I don't have enough information in the provided context to answer this question."
- Be concise and accurate
- Cite sources when providing information by referencing the relevant parts of the context
- Do not make up information or go beyond what's in the context
- Maintain the tone appropriate for an educational resource

QUESTION:
{query}

ANSWER:"""

        return prompt, sources

    def build_grounding_check_prompt(self, answer: str, context: str) -> str:
        """
        Build a prompt to check if the answer is properly grounded in the context
        """
        prompt = f"""Check if the following answer is properly grounded in the provided context.

CONTEXT:
{context}

ANSWER:
{answer}

ASSESSMENT:
- Is the answer factually consistent with the context? (Yes/No)
- Does the answer contain information not present in the context? (Yes/No)
- Are any claims made without support from the context? (Yes/No)

FEEDBACK:
Provide specific feedback on any inconsistencies or unsupported claims."""

        return prompt

    def build_citation_prompt(self, answer: str, sources: List[str]) -> str:
        """
        Build a prompt to properly format citations in the answer
        """
        prompt = f"""Review the following answer and ensure proper citation format.

SOURCES:
{', '.join(sources)}

ANSWER BEFORE CITATION FORMATTING:
{answer}

INSTRUCTIONS:
- Add proper citations to the answer using the source information
- Use the format [SOURCE: source_reference] after relevant statements
- Ensure all information derived from sources is properly cited
- Do not alter the content of the answer, only add citations

ANSWER WITH CITATIONS:"""

        return prompt

    def apply_grounding_instructions(self, query: str, context: str, selected_context: str = None) -> str:
        """
        Apply grounding instructions to a query with context
        """
        if selected_context:
            full_context = f"Selected Text Context:\n{selected_context}\n\nDocument Context:\n{context}"
        else:
            full_context = context

        prompt = f"""{full_context}

Based on the above context, please answer the following question with strict adherence to the information provided:

Question: {query}

Answer (based ONLY on the provided context):"""

        return prompt

# Create a singleton instance
chat_prompt_engineer = ChatPromptEngineer()