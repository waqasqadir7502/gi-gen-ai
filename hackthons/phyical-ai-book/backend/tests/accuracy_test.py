import json
import asyncio
from typing import List, Dict, Tuple
from ..services.retrieval_service import retrieval_service
from ..services.generation_service import generation_service
from ..utils.logger import log_info, log_error
from ..config import config

class AccuracyValidator:
    def __init__(self):
        self.test_questions = []
        self.min_accuracy_threshold = 0.94  # 94%
        self.max_hallucination_rate = 0.06  # 6%

    def load_test_questions(self, filepath: str = "test_data/questions.json"):
        """
        Load test questions from a JSON file
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                self.test_questions = json.load(f)
            log_info(f"Loaded {len(self.test_questions)} test questions", extra={
                "filepath": filepath
            })
            return True
        except FileNotFoundError:
            log_error(f"Test questions file not found: {filepath}")
            # Use default test questions
            self.test_questions = [
                {
                    "question": "What is the Physical AI Book about?",
                    "expected_sources": ["README.md", "docs/chapter-template.md"],
                    "category": "general"
                },
                {
                    "question": "How is the content structured?",
                    "expected_sources": ["docs/content-standards.md"],
                    "category": "structure"
                }
            ]
            log_info(f"Using default test questions: {len(self.test_questions)}")
            return False
        except Exception as e:
            log_error(f"Error loading test questions: {str(e)}")
            return False

    def evaluate_answer_accuracy(self, question: str, answer: str, expected_sources: List[str]) -> Tuple[bool, Dict]:
        """
        Evaluate if the answer is accurate based on expected sources
        """
        # Check if the answer contains relevant information
        answer_lower = answer.lower()

        # Check for relevant keywords from expected sources
        relevant_keywords_found = 0
        total_expected_keywords = 0

        for source in expected_sources:
            # In a real implementation, we'd check actual content from the source
            # For now, we'll just check if source is referenced
            if any(keyword in answer_lower for keyword in source.lower().split('/')):
                relevant_keywords_found += 1
            total_expected_keywords += 1

        accuracy_score = relevant_keywords_found / max(total_expected_keywords, 1)

        evaluation = {
            "accuracy_score": accuracy_score,
            "relevant_sources_found": relevant_keywords_found,
            "total_expected_sources": total_expected_keywords,
            "is_accurate": accuracy_score >= self.min_accuracy_threshold
        }

        return evaluation["is_accurate"], evaluation

    def measure_hallucination_rate(self, answer: str, retrieved_chunks: List[Dict]) -> float:
        """
        Measure the hallucination rate in the answer
        """
        # This is a simplified check - in practice, you'd use more sophisticated NLP methods
        answer_lower = answer.lower()

        # Count how much of the answer is supported by retrieved content
        total_answer_length = len(answer)
        supported_length = 0

        for chunk in retrieved_chunks:
            chunk_content = chunk.get('content', '').lower()
            # Find overlapping content between answer and retrieved chunks
            if chunk_content in answer_lower:
                supported_length += len(chunk_content)

        # Calculate hallucination rate
        if total_answer_length > 0:
            supported_ratio = min(supported_length / total_answer_length, 1.0)
            hallucination_rate = 1.0 - supported_ratio
        else:
            hallucination_rate = 1.0

        return min(hallucination_rate, 1.0)  # Cap at 1.0

    async def run_comprehensive_accuracy_test(self, sample_size: int = None) -> Dict:
        """
        Run comprehensive accuracy tests on the RAG pipeline
        """
        if not self.test_questions:
            self.load_test_questions()

        if sample_size:
            test_questions = self.test_questions[:sample_size]
        else:
            test_questions = self.test_questions

        results = {
            "total_tests": len(test_questions),
            "passed": 0,
            "failed": 0,
            "accuracy_rate": 0,
            "hallucination_rate": 0,
            "detailed_results": []
        }

        total_hallucination_rate = 0

        for i, test_case in enumerate(test_questions):
            question = test_case.get("question", "")
            expected_sources = test_case.get("expected_sources", [])

            try:
                # Retrieve relevant chunks
                retrieved_chunks = retrieval_service.retrieve_and_rerank(
                    query=question,
                    top_k=config.TOP_K
                )

                # Generate response
                generation_result = generation_service.generate_response(
                    query=question,
                    context_chunks=retrieved_chunks
                )

                answer = generation_result["answer"]

                # Evaluate accuracy
                is_accurate, accuracy_eval = self.evaluate_answer_accuracy(
                    question, answer, expected_sources
                )

                # Measure hallucination rate
                hallucination_rate = self.measure_hallucination_rate(answer, retrieved_chunks)
                total_hallucination_rate += hallucination_rate

                # Record result
                test_result = {
                    "test_id": i,
                    "question": question,
                    "is_accurate": is_accurate,
                    "accuracy_score": accuracy_eval["accuracy_score"],
                    "hallucination_rate": hallucination_rate,
                    "sources_found": accuracy_eval["relevant_sources_found"],
                    "expected_sources": len(expected_sources),
                    "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer
                }

                results["detailed_results"].append(test_result)

                if is_accurate and hallucination_rate <= self.max_hallucination_rate:
                    results["passed"] += 1
                else:
                    results["failed"] += 1

                log_info(f"Test {i+1}/{len(test_questions)} completed", extra={
                    "question": question[:50] + "..." if len(question) > 50 else question,
                    "accurate": is_accurate,
                    "hallucination_rate": hallucination_rate
                })

            except Exception as e:
                log_error(f"Error running test {i+1}: {str(e)}", extra={
                    "question": question[:50] + "..." if len(question) > 50 else question
                })
                results["failed"] += 1

                test_result = {
                    "test_id": i,
                    "question": question,
                    "is_accurate": False,
                    "accuracy_score": 0,
                    "hallucination_rate": 1.0,
                    "error": str(e)
                }
                results["detailed_results"].append(test_result)

        # Calculate overall metrics
        if results["total_tests"] > 0:
            results["accuracy_rate"] = results["passed"] / results["total_tests"]
            results["hallucination_rate"] = total_hallucination_rate / results["total_tests"]

        log_info("Comprehensive accuracy test completed", extra={
            "total_tests": results["total_tests"],
            "passed": results["passed"],
            "failed": results["failed"],
            "accuracy_rate": results["accuracy_rate"],
            "overall_hallucination_rate": results["hallucination_rate"]
        })

        return results

    def validate_accuracy_target(self, results: Dict) -> bool:
        """
        Validate if the accuracy meets the target threshold
        """
        return (
            results["accuracy_rate"] >= self.min_accuracy_threshold and
            results["hallucination_rate"] <= self.max_hallucination_rate
        )

# Create a singleton instance
accuracy_validator = AccuracyValidator()