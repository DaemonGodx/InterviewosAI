from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import GROQ_API_KEY


class Evaluator:

    def __init__(self, role, experience):

        self.role = role
        self.experience = experience

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=GROQ_API_KEY,
            temperature=0.2
        )

    def evaluate(self, questions, answers):

        interview_data = ""

        for i, (question, answer) in enumerate(
            zip(questions, answers), 1
        ):
            interview_data += f"""
Question {i}:
{question}

Candidate Answer:
{answer}

"""

        prompt = f"""
You are a strict but fair technical interviewer.

Evaluate the following candidate interview.

Candidate Role: {self.role}
Experience Level: {self.experience}

{interview_data}

Evaluate the candidate based ONLY on their answers.

Provide the evaluation in exactly this format:

FINAL EVALUATION

Overall Score: X/10

Technical Knowledge: X/10
Problem Solving: X/10
Communication: X/10

Strengths:
- strength 1
- strength 2
- strength 3

Weaknesses:
- weakness 1
- weakness 2
- weakness 3

Technical Assessment:
Write a short assessment of the candidate's technical understanding.

Recommendations:
- recommendation 1
- recommendation 2
- recommendation 3

Be honest.
Do not give a high score just to be encouraging.
"""

        response = self.llm.invoke([
            SystemMessage(
                content="You are an expert technical interviewer."
            ),
            HumanMessage(content=prompt)
        ])

        return response.content