def generate_questions(role, level):
    return f"""
Instruction:
Generate exactly 5 interview questions.

Role: {role}
Level: {level}

Format STRICTLY like this:
1. Question one?
2. Question two?
3. Question three?
4. Question four?
5. Question five?

Do not add anything else.
"""

def evaluate_answer(role, question, answer):
    return f"""
Instruction:
You are a {role} interviewer.

Task:
Evaluate the candidate answer.

Question:
{question}

Answer:
{answer}

Output:
Score: <number>/10
Strengths:
- ...
Weaknesses:
- ...
"""

def improve_answer(role, answer, feedback):
    return f"""
Instruction:
Improve the answer professionally.

Role:
{role}

Answer:
{answer}

Feedback:
{feedback}

Output:
Improved answer only
"""

def final_report(role, scores):
    return f"""
Instruction:
Generate final interview report.

Role:
{role}

Scores:
{scores}

Output:
- Overall Rating
- Strengths
- Weak Areas
- Hiring Decision
"""