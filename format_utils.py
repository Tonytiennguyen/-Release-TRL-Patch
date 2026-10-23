import re

# The exact system prompt used to elicit formatting from the Instruct model
SYSTEM_PROMPT = """
You are a mathematical reasoning assistant. 
You must solve the problem step-by-step. 
CRITICAL INSTRUCTION: You must enclose your final numerical answer exactly inside the following tags: <answer>YOUR_NUMBER</answer>. 
Do not add any additional text inside the tags.
"""

def extract_answer(generation_text: str) -> str:
    """
    Regex parser used in the reward function to capture the formatted answer.
    """
    match = re.search(r'<answer>(.*?)</answer>', generation_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def compute_format_reward(generation_text: str, reference_answer: str) -> float:
    """
    Assigns a baseline reward for formatting compliance, ensuring non-zero gradients.
    """
    extracted = extract_answer(generation_text)
    if extracted is None:
        return 0.0 # Failed format
    elif extracted == reference_answer:
        return 1.0 # Correct and formatted
    else:
        return 0.1 # Formatted correctly, but wrong answer (partial signal)
