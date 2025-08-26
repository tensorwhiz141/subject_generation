"""
Quiz Evaluation Module for Gurukul Learning Platform
Handles quiz submission, scoring, and performance analysis
"""

import json
import re
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from difflib import SequenceMatcher

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuizEvaluator:
    """Evaluates quiz submissions and provides detailed feedback"""
    
    def __init__(self):
        self.similarity_threshold = 0.7  # For short answer evaluation
        
    def evaluate_quiz_submission(
        self, 
        quiz_data: Dict[str, Any], 
        user_answers: Dict[str, Any],
        user_id: str = "anonymous"
    ) -> Dict[str, Any]:
        """
        Evaluate a complete quiz submission
        
        Args:
            quiz_data: Original quiz data with questions and correct answers
            user_answers: User's submitted answers
            user_id: User identifier for tracking
            
        Returns:
            Dict containing detailed evaluation results
        """
        try:
            logger.info(f"Evaluating quiz submission for user {user_id}")
            
            questions = quiz_data.get("questions", [])
            total_questions = len(questions)
            correct_answers = 0
            total_points = 0
            max_points = quiz_data.get("scoring", {}).get("total_points", total_questions * 10)
            
            detailed_results = []
            
            # Evaluate each question
            for i, question in enumerate(questions):
                question_id = question.get("question_id", f"q_{i+1}")
                user_answer = user_answers.get(question_id)
                
                result = self._evaluate_single_question(question, user_answer)
                detailed_results.append(result)
                
                if result["is_correct"]:
                    correct_answers += 1
                
                total_points += result["points_earned"]
            
            # Calculate performance metrics
            percentage_score = (total_points / max_points) * 100 if max_points > 0 else 0
            passing_score = quiz_data.get("scoring", {}).get("passing_score", max_points * 0.6)
            passed = total_points >= passing_score
            
            # Generate performance analysis
            performance_analysis = self._generate_performance_analysis(
                detailed_results, 
                percentage_score, 
                quiz_data.get("subject", ""), 
                quiz_data.get("topic", "")
            )
            
            # Create evaluation result
            evaluation_result = {
                "evaluation_id": f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "user_id": user_id,
                "quiz_id": quiz_data.get("quiz_id", "unknown"),
                "subject": quiz_data.get("subject", ""),
                "topic": quiz_data.get("topic", ""),
                "submitted_at": datetime.now().isoformat(),
                
                # Scoring summary
                "score_summary": {
                    "total_questions": total_questions,
                    "correct_answers": correct_answers,
                    "incorrect_answers": total_questions - correct_answers,
                    "total_points": total_points,
                    "max_points": max_points,
                    "percentage_score": round(percentage_score, 2),
                    "passed": passed,
                    "grade": self._calculate_grade(percentage_score)
                },
                
                # Detailed question-by-question results
                "detailed_results": detailed_results,
                
                # Performance analysis and recommendations
                "performance_analysis": performance_analysis,
                
                # Time and completion data
                "completion_data": {
                    "estimated_time": quiz_data.get("estimated_time", 0),
                    "difficulty": quiz_data.get("difficulty", "medium")
                }
            }
            
            logger.info(f"Quiz evaluation completed: {correct_answers}/{total_questions} correct ({percentage_score:.1f}%)")
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating quiz submission: {e}")
            return self._generate_error_result(user_id, str(e))
    
    def _evaluate_single_question(self, question: Dict[str, Any], user_answer: Any) -> Dict[str, Any]:
        """Evaluate a single question based on its type"""
        try:
            question_type = question.get("type", "multiple_choice")
            question_id = question.get("question_id", "unknown")
            max_points = question.get("points", 10)
            
            if question_type == "multiple_choice":
                return self._evaluate_multiple_choice(question, user_answer, max_points)
            elif question_type == "true_false":
                return self._evaluate_true_false(question, user_answer, max_points)
            elif question_type == "fill_in_blank":
                return self._evaluate_fill_in_blank(question, user_answer, max_points)
            elif question_type == "short_answer":
                return self._evaluate_short_answer(question, user_answer, max_points)
            else:
                return self._generate_question_error_result(question_id, "Unknown question type", max_points)
                
        except Exception as e:
            logger.error(f"Error evaluating question {question.get('question_id', 'unknown')}: {e}")
            return self._generate_question_error_result(question.get("question_id", "unknown"), str(e), 10)
    
    def _evaluate_multiple_choice(self, question: Dict[str, Any], user_answer: Any, max_points: int) -> Dict[str, Any]:
        """Evaluate multiple choice question"""
        correct_answer = question.get("correct_answer", 0)
        is_correct = user_answer == correct_answer
        
        return {
            "question_id": question.get("question_id"),
            "question_type": "multiple_choice",
            "question_text": question.get("question"),
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "points_earned": max_points if is_correct else 0,
            "max_points": max_points,
            "feedback": self._generate_feedback(question, is_correct),
            "explanation": question.get("explanation", "")
        }
    
    def _evaluate_true_false(self, question: Dict[str, Any], user_answer: Any, max_points: int) -> Dict[str, Any]:
        """Evaluate true/false question"""
        correct_answer = question.get("correct_answer", True)
        is_correct = user_answer == correct_answer
        
        return {
            "question_id": question.get("question_id"),
            "question_type": "true_false",
            "question_text": question.get("question"),
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
            "points_earned": max_points if is_correct else 0,
            "max_points": max_points,
            "feedback": self._generate_feedback(question, is_correct),
            "explanation": question.get("explanation", "")
        }
    
    def _evaluate_fill_in_blank(self, question: Dict[str, Any], user_answer: Any, max_points: int) -> Dict[str, Any]:
        """Evaluate fill-in-the-blank question"""
        correct_answer = question.get("correct_answer", "").strip().lower()
        user_answer_clean = str(user_answer).strip().lower() if user_answer else ""
        
        # Check for exact match or high similarity
        is_exact_match = user_answer_clean == correct_answer
        similarity = SequenceMatcher(None, user_answer_clean, correct_answer).ratio()
        is_similar = similarity >= self.similarity_threshold
        
        is_correct = is_exact_match or is_similar
        points_earned = max_points if is_exact_match else (max_points * 0.8 if is_similar else 0)
        
        return {
            "question_id": question.get("question_id"),
            "question_type": "fill_in_blank",
            "question_text": question.get("question"),
            "user_answer": user_answer,
            "correct_answer": question.get("correct_answer"),
            "is_correct": is_correct,
            "points_earned": int(points_earned),
            "max_points": max_points,
            "similarity_score": round(similarity, 2),
            "feedback": self._generate_feedback(question, is_correct),
            "explanation": question.get("explanation", "")
        }
    
    def _evaluate_short_answer(self, question: Dict[str, Any], user_answer: Any, max_points: int) -> Dict[str, Any]:
        """Evaluate short answer question using similarity matching"""
        sample_answer = question.get("sample_answer", "").strip().lower()
        user_answer_clean = str(user_answer).strip().lower() if user_answer else ""
        
        if not user_answer_clean:
            similarity = 0.0
            is_correct = False
            points_earned = 0
        else:
            # Calculate similarity with sample answer
            similarity = SequenceMatcher(None, user_answer_clean, sample_answer).ratio()
            
            # Check for key terms presence
            key_terms = self._extract_key_terms(sample_answer)
            term_matches = sum(1 for term in key_terms if term in user_answer_clean)
            term_score = term_matches / len(key_terms) if key_terms else 0
            
            # Combined scoring
            combined_score = (similarity * 0.6) + (term_score * 0.4)
            is_correct = combined_score >= 0.5
            points_earned = int(max_points * combined_score)
        
        return {
            "question_id": question.get("question_id"),
            "question_type": "short_answer",
            "question_text": question.get("question"),
            "user_answer": user_answer,
            "sample_answer": question.get("sample_answer"),
            "is_correct": is_correct,
            "points_earned": points_earned,
            "max_points": max_points,
            "similarity_score": round(similarity, 2),
            "feedback": self._generate_feedback(question, is_correct),
            "explanation": question.get("explanation", "")
        }
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for evaluation"""
        # Remove common words and extract meaningful terms
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should"}
        words = re.findall(r'\b\w+\b', text.lower())
        key_terms = [word for word in words if word not in common_words and len(word) > 2]
        return list(set(key_terms))
    
    def _generate_feedback(self, question: Dict[str, Any], is_correct: bool) -> str:
        """Generate personalized feedback for the question"""
        if is_correct:
            return "Excellent! Your answer is correct."
        else:
            return f"Not quite right. {question.get('explanation', 'Please review the material and try again.')}"
    
    def _calculate_grade(self, percentage: float) -> str:
        """Calculate letter grade based on percentage"""
        if percentage >= 90:
            return "A"
        elif percentage >= 80:
            return "B"
        elif percentage >= 70:
            return "C"
        elif percentage >= 60:
            return "D"
        else:
            return "F"
    
    def _generate_performance_analysis(self, results: List[Dict], percentage: float, subject: str, topic: str) -> Dict[str, Any]:
        """Generate detailed performance analysis and recommendations"""
        total_questions = len(results)
        correct_count = sum(1 for r in results if r["is_correct"])
        
        # Analyze performance by question type
        type_performance = {}
        for result in results:
            q_type = result["question_type"]
            if q_type not in type_performance:
                type_performance[q_type] = {"correct": 0, "total": 0}
            type_performance[q_type]["total"] += 1
            if result["is_correct"]:
                type_performance[q_type]["correct"] += 1
        
        # Generate recommendations
        recommendations = []
        if percentage < 60:
            recommendations.append(f"Consider reviewing the fundamental concepts of {topic} in {subject}")
            recommendations.append("Practice more questions to strengthen your understanding")
        elif percentage < 80:
            recommendations.append(f"Good progress! Focus on the areas where you missed questions")
            recommendations.append("Try more advanced practice questions")
        else:
            recommendations.append("Excellent performance! You have a strong understanding of the topic")
            recommendations.append("Consider exploring more advanced topics in this subject")
        
        return {
            "overall_performance": "Excellent" if percentage >= 80 else "Good" if percentage >= 60 else "Needs Improvement",
            "strengths": [f"Strong performance in {q_type}" for q_type, perf in type_performance.items() if perf["correct"]/perf["total"] >= 0.8],
            "areas_for_improvement": [f"Review {q_type} questions" for q_type, perf in type_performance.items() if perf["correct"]/perf["total"] < 0.6],
            "recommendations": recommendations,
            "type_performance": type_performance
        }
    
    def _generate_question_error_result(self, question_id: str, error_msg: str, max_points: int) -> Dict[str, Any]:
        """Generate error result for a question"""
        return {
            "question_id": question_id,
            "question_type": "error",
            "error": error_msg,
            "is_correct": False,
            "points_earned": 0,
            "max_points": max_points,
            "feedback": "There was an error evaluating this question."
        }
    
    def _generate_error_result(self, user_id: str, error_msg: str) -> Dict[str, Any]:
        """Generate error result for entire quiz evaluation"""
        return {
            "evaluation_id": f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "user_id": user_id,
            "error": error_msg,
            "submitted_at": datetime.now().isoformat(),
            "score_summary": {
                "total_questions": 0,
                "correct_answers": 0,
                "total_points": 0,
                "max_points": 0,
                "percentage_score": 0,
                "passed": False,
                "grade": "F"
            }
        }
