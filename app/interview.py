from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from app.config import GROQ_API_KEY
from app.prompts import SYSTEM_PROMPT
from app.analyzer import AnswerAnalyzer


class Interviewer:

    def __init__(self, role, experience):

        self.role = role
        self.experience = experience

        self.llm = ChatGroq(
            model="openai/gpt-oss-120b",
            groq_api_key=GROQ_API_KEY,
            temperature=0.7
        )

        self.analyzer = AnswerAnalyzer()

        self.messages = [
            SystemMessage(
                content=SYSTEM_PROMPT.format(
                    role=role,
                    experience=experience
                )
            )
        ]

        self.questions = []
        self.answers = []
        self.difficulties = []
        self.answer_levels = []

        self.current_difficulty = "MEDIUM"

    def start(self):

        prompt = f"""
Start the interview.

Candidate role: {self.role}

Ask the first {self.current_difficulty} difficulty question.
Ask ONLY the question.
"""

        response = self.llm.invoke([
            *self.messages,
            HumanMessage(content=prompt)
        ])

        question = response.content

        self.questions.append(question)
        self.difficulties.append(self.current_difficulty)

        self.messages.append(response)

        return question

    def ask(self, answer):

        current_question = self.questions[-1]

        # Store answer
        self.answers.append(answer)

        # Analyze answer
        level = self.analyzer.analyze(
            current_question,
            answer
        )

        self.answer_levels.append(level)

        # Adjust difficulty
        if level == "STRONG":

            if self.current_difficulty == "EASY":
                self.current_difficulty = "MEDIUM"

            elif self.current_difficulty == "MEDIUM":
                self.current_difficulty = "HARD"

        elif level == "WEAK":

            if self.current_difficulty == "HARD":
                self.current_difficulty = "MEDIUM"

            elif self.current_difficulty == "MEDIUM":
                self.current_difficulty = "EASY"

        # Generate next question
        prompt = f"""
The candidate just answered the previous question.

Candidate answer quality:
{level}

Previous question:
{current_question}

Generate the next interview question.

Current difficulty:
{self.current_difficulty}

Candidate role:
{self.role}

Requirements:

- Ask ONLY ONE question.
- Do not repeat the previous question.
- Make the question appropriate for the current difficulty.
- If the previous answer was WEAK, test the underlying concept more simply.
- If the previous answer was AVERAGE, explore the concept further.
- If the previous answer was STRONG, ask a deeper question.
"""

        self.messages.append(
            HumanMessage(content=answer)
        )

        response = self.llm.invoke(
            self.messages + [
                HumanMessage(content=prompt)
            ]
        )

        question = response.content

        self.questions.append(question)
        self.difficulties.append(self.current_difficulty)

        self.messages.append(response)

        return question

    def get_questions(self):
        return self.questions

    def get_answers(self):
        return self.answers

    def get_difficulties(self):
        return self.difficulties

    def get_answer_levels(self):
        return self.answer_levels