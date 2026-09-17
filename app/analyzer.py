from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

from app.config import GROQ_API_KEY


class AnswerAnalyzer:

    def __init__(self):

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=GROQ_API_KEY,
            temperature=0.1
        )

    def analyze(self, question, answer):

        prompt = f"""
Analyze this technical interview answer.

Question:
{question}

Candidate Answer:
{answer}

Classify the answer into exactly ONE category:

WEAK
AVERAGE
STRONG

Definitions:

WEAK:
- Incorrect or mostly incorrect
- Very vague
- Shows little understanding
- Cannot explain the concept

AVERAGE:
- Partially correct
- Understands the basic concept
- Missing depth or important details

STRONG:
- Correct
- Clear explanation
- Demonstrates practical understanding
- Can explain reasoning or trade-offs

Return ONLY one word:

WEAK
or
AVERAGE
or
STRONG
"""

        response = self.llm.invoke([
            HumanMessage(content=prompt)
        ])

        result = response.content.strip().upper()

        if "STRONG" in result:
            return "STRONG"

        if "AVERAGE" in result:
            return "AVERAGE"

        return "WEAK"