from app.interview import Interviewer
from app.evaluator import Evaluator


def main():

    print("=" * 50)
    print("        INTERVIEWOS")
    print("=" * 50)

    role = input("\nEnter your role: ")
    experience = input("Enter your experience level: ")

    interviewer = Interviewer(
        role=role,
        experience=experience
    )

    print("\nInterview started!\n")

    question = interviewer.start()

    print("Interviewer:")
    print(question)

    question_count = 1

    while question_count <= 5:

        print("\nYou:")
        answer = input("> ")

        if answer.lower() in ["exit", "quit"]:
            print("\nInterview ended.")
            return

        question = interviewer.ask(answer)

        # Show internal adaptive decision
        level = interviewer.get_answer_levels()[-1]
        difficulty = interviewer.get_difficulties()[-1]

        print(f"\n[Answer Analysis: {level}]")
        print(f"[Next Difficulty: {difficulty}]")

        if question_count < 5:
            print("\nInterviewer:")
            print(question)

        question_count += 1

    print("\n" + "=" * 50)
    print("       EVALUATING INTERVIEW")
    print("=" * 50)

    evaluator = Evaluator(
        role=role,
        experience=experience
    )

    evaluation = evaluator.evaluate(
        interviewer.get_questions()[:-1],
        interviewer.get_answers()
    )

    print("\n")
    print(evaluation)

    print("\n" + "=" * 50)
    print("        INTERVIEW COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()