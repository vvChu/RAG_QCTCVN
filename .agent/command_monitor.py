"""
Command Monitoring & Completion Time Estimation

This module helps estimate command completion time to optimize `command_status` polling.
"""

import re
from typing import Optional, Dict, Tuple

class CommandTimeEstimator:
    """Estimates command completion time based on output patterns."""
    
    # Base waiting times for different operations (in seconds)
    OPERATION_TIMES = {
        "model_loading": 120,      # Loading BGE-M3, transformers
        "embedding_per_chunk": 0.15,  # ~150ms per chunk
        "parsing_pdf": 30,         # PDF parsing
        "parsing_docx": 15,        # DOCX parsing
        "milvus_insert": 10,       # Insertion overhead
        "evaluation_per_question": 8,  # Per question in eval
        "default": 60,
    }
    
    # Patterns to detect specific operations
    PATTERNS = {
        "model_loading": [
            r"Loading [\w\-]+ model",
            r"Fetching \d+ files:",
        ],
        "embedding": [
            r"Embedding (\d+) chunks",
        ],
        "parsing": [
            r"Processing ([\w\-\.]+\.pdf)",
            r"Processing ([\w\-\.]+\.docx)",
        ],
        "evaluation": [
            r"Evaluating:",
            r"Total Questions: (\d+)",
        ],
        "insertion": [
            r"Inserting (\d+) chunks into Milvus",
        ],
    }
    
    def __init__(self):
        self.last_estimate = 60
        self.check_count = 0
    
    def estimate_wait_time(self, command: str, current_output: str) -> int:
        """
        Estimate how long to wait before checking command status again.
        
        Args:
            command: The command being executed
            current_output: Current output from the command
            
        Returns:
            Recommended wait duration in seconds
        """
        self.check_count += 1
        
        # Check for specific patterns in output
        estimates = []
        
        # Model loading detected
        if self._matches_any(current_output, self.PATTERNS["model_loading"]):
            estimates.append(self.OPERATION_TIMES["model_loading"])
        
        # Embedding detected
        embedding_match = self._extract_number(current_output, self.PATTERNS["embedding"])
        if embedding_match:
            num_chunks = int(embedding_match)
            # Estimate: num_chunks * time_per_chunk + overhead
            time_estimate = num_chunks * self.OPERATION_TIMES["embedding_per_chunk"] + 20
            estimates.append(min(time_estimate, 300))  # Cap at 5 minutes
        
        # Parsing detected
        if self._matches_any(current_output, self.PATTERNS["parsing"]):
            if ".pdf" in current_output:
                estimates.append(self.OPERATION_TIMES["parsing_pdf"])
            elif ".docx" in current_output:
                estimates.append(self.OPERATION_TIMES["parsing_docx"])
        
        # Insertion detected
        insertion_match = self._extract_number(current_output, self.PATTERNS["insertion"])
        if insertion_match:
            estimates.append(self.OPERATION_TIMES["milvus_insert"])
        
        # Evaluation detected
        eval_match = self._extract_number(current_output, self.PATTERNS["evaluation"])
        if eval_match:
            num_questions = int(eval_match)
            estimates.append(num_questions * self.OPERATION_TIMES["evaluation_per_question"])
        
        # If no specific pattern matched, use adaptive logic
        if not estimates:
            # Adaptive waiting: increase wait time after multiple "no output" checks
            if "No new output" in current_output or not current_output.strip():
                # Progressive backoff
                if self.check_count == 1:
                    return 60
                elif self.check_count == 2:
                    return 120
                elif self.check_count >= 3:
                    return 180
            return self.OPERATION_TIMES["default"]
        
        # Use the maximum estimate (most conservative)
        wait_time = max(estimates)
        self.last_estimate = wait_time
        
        return int(wait_time)
    
    def reset(self):
        """Reset counter for a new command."""
        self.check_count = 0
        self.last_estimate = 60
    
    def _matches_any(self, text: str, patterns: list) -> bool:
        """Check if text matches any of the patterns."""
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _extract_number(self, text: str, patterns: list) -> Optional[str]:
        """Extract number from text using patterns."""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.groups():
                return match.group(1)
        return None


# Global estimator instance
estimator = CommandTimeEstimator()


def get_optimal_wait_time(command: str, output: str) -> int:
    """
    Get optimal wait time for command status check.
    
    Usage in agent code:
        from .agent.command_monitor import get_optimal_wait_time
        
        wait_time = get_optimal_wait_time(command_line, current_output)
        command_status(CommandId=cmd_id, WaitDurationSeconds=wait_time)
    
    Args:
        command: The command being executed
        output: Current command output
        
    Returns:
        Optimal wait duration in seconds
    """
    return estimator.estimate_wait_time(command, output)


def reset_estimator():
    """Reset estimator for a new command."""
    estimator.reset()


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("indexing", "Loading BGE-M3 model: BAAI/bge-m3", "Should wait ~120s for model loading"),
        ("indexing", "Embedding 781 chunks...", "Should wait ~137s for 781 chunks"),
        ("indexing", "Processing document.pdf...", "Should wait ~30s for PDF parsing"),
        ("eval", "Total Questions: 5", "Should wait ~40s for 5 questions"),
        ("indexing", "No new output since last check", "Should progressively increase"),
    ]
    
    print("=== Command Time Estimator Test ===\n")
    for cmd, output, expected in test_cases:
        estimator.reset()
        wait_time = get_optimal_wait_time(cmd, output)
        print(f"Command: {cmd}")
        print(f"Output: {output}")
        print(f"Estimated wait: {wait_time}s")
        print(f"Expected: {expected}")
        print()
