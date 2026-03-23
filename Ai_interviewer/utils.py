import re

def parse_questions(text):
    """
    Extract numbered questions from LLM output
    """

    # Try to extract numbered questions (1. 2. 3. format)
    questions = re.findall(r'\d+\.\s*(.*)', text)

    # If regex fails → fallback
    if not questions:
        questions = [line.strip() for line in text.split("\n") if line.strip()]

    return questions


def extract_score(feedback):
    """
    Extract score from feedback text
    """

    match = re.search(r'(\d+)/10', feedback)

    if match:
        return int(match.group(1))

    # fallback (any number)
    match = re.search(r'(\d+)', feedback)
    if match:
        return min(int(match.group(1)), 10)

    return 5