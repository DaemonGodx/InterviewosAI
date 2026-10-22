import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.nodes import generate_question_node

def main():
    state = {
        "session_id": "test",
        "config": {
            "interview_type": "Mixed",
            "question_count": 5,
            "target_role": "Backend Developer",
            "experience_level": "Mid"
        },
        "profile": {
            "resume_summary": "A backend developer with 3 years of experience in Python, Django, and PostgreSQL.",
            "interviewable_technologies": ["Python", "Django", "PostgreSQL", "Redis"]
        },
        "questions": [],
        "topics_covered": [],
        "current_difficulty": "Medium",
        "question_metadata": []
    }

    for i in range(5):
        res = generate_question_node(state)
        state.update(res)
        q_idx = len(state["questions"])
        print(f"\nQuestion {q_idx} / 5")
        print(f"Type: {res['current_question_type']}")
        print(f"Question: {res['current_question']}".encode('utf-8', 'replace').decode('utf-8'))
        print("-" * 40)

if __name__ == "__main__":
    main()
