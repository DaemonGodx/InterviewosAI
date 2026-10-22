import sys
import os
import uuid
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from rich.table import Table
from rich.text import Text

from app.config import validate_config
from app.models.db import Database
from app.services.resume_service import ResumeService, ResumeExtractionError
from app.models.schemas import InterviewConfig, InterviewState, CandidateProfile
from app.engine.graph import InterviewEngine
from app.services.planner_service import PlannerService
from app.audio.audio_service import AudioService, AudioServiceError
from app.services.speech_service import SpeechService, SpeechServiceError

console = Console()

def show_header():
    header_text = """
╔══════════════════════════════════════════╗
║              INTERVIEWOS                 ║
║       AI Interview Preparation           ║
╚══════════════════════════════════════════╝
"""
    console.print(header_text, style="bold blue")

def get_resume_path() -> str:
    while True:
        path = Prompt.ask("\n[bold]Resume path[/bold] (PDF or DOCX)")
        path = path.strip().strip("'\"")
        if not path:
            console.print("[red]Path cannot be empty.[/red]")
            continue
        if not os.path.exists(path):
            console.print(f"[red]File not found: {path}[/red]")
            continue
        if not (path.lower().endswith(".pdf") or path.lower().endswith(".docx")):
            console.print("[red]Only PDF and DOCX formats are supported.[/red]")
            continue
        return path

def display_profile(profile: CandidateProfile):
    console.print("\n[bold green]✓ Resume parsed[/bold green]")
    console.print("[bold green]✓ Candidate profile generated[/bold green]\n")
    
    table = Table(title="Candidate Profile Summary", show_header=False, box=None)
    table.add_column("Key", style="bold cyan")
    table.add_column("Value")
    
    table.add_row("Name:", profile.name)
    table.add_row("Education:", ", ".join(profile.education) if profile.education else "N/A")
    table.add_row("Skills:", ", ".join(profile.skills) if profile.skills else "N/A")
    table.add_row("Experience:", str(len(profile.experience)) + " roles listed")
    table.add_row("Projects:", str(len(profile.projects)) + " projects listed")
    table.add_row("Summary:", profile.resume_summary or "N/A")
    table.add_row("Inferred Strengths:", ", ".join(profile.inferred_strengths) if profile.inferred_strengths else "None")
    
    estimated_exp = f"~{profile.estimated_years_experience} years" if profile.estimated_years_experience is not None else "Unknown"
    recommended_level = profile.recommended_experience_level or "Unknown"
    
    console.print(Panel(table, expand=False))
    console.print(f"\n[bold]Detected experience:[/bold] {estimated_exp}")
    console.print(f"[bold]Recommended experience level:[/bold] {recommended_level}")

def configure_interview(profile: CandidateProfile) -> InterviewConfig:
    console.print("\n[bold]Select target role:[/bold]")
    roles = ["Backend Developer", "Full Stack Developer", "Frontend Developer", "Data Scientist", "AI/ML Engineer", "Custom role"]
    for i, role in enumerate(roles, 1):
        console.print(f"{i}. {role}")
        
    role_choice = IntPrompt.ask("Select", choices=[str(i) for i in range(1, len(roles) + 1)])
    if role_choice == len(roles):
        target_role = Prompt.ask("Enter custom target role")
    else:
        target_role = roles[role_choice - 1]

    console.print("\n[bold]Select experience level:[/bold]")
    exp_levels = ["Student", "Fresher", "0-2 years", "2-5 years", "5+ years"]
    
    recommended = profile.recommended_experience_level
    for i, level in enumerate(exp_levels, 1):
        if recommended and level == recommended:
            console.print(f"{i}. {level} [green](Recommended)[/green]")
        else:
            console.print(f"{i}. {level}")
            
    exp_choice = IntPrompt.ask("Select", choices=[str(i) for i in range(1, len(exp_levels) + 1)])
    experience_level = exp_levels[exp_choice - 1]
    
    # Check mismatch
    if recommended and experience_level != recommended:
        console.print(f"\n[bold yellow]⚠ Experience mismatch detected[/bold yellow]")
        console.print(f"Your resume indicates approximately {profile.estimated_years_experience} years of professional experience, but you selected '{experience_level}'.")
        console.print("This may make the interview difficulty and evaluation inconsistent with your resume.")
        proceed = Prompt.ask("Continue with selected level?", choices=["y", "n"], default="y")
        if proceed.lower() == 'n':
            return configure_interview(profile)
    
    console.print("\n[bold]Interview type:[/bold]")
    types = ["Technical", "Behavioral", "Mixed"]
    for i, t in enumerate(types, 1):
        console.print(f"{i}. {t}")
    type_choice = IntPrompt.ask("Select", choices=["1", "2", "3"])
    interview_type = types[type_choice - 1]
    
    while True:
        question_count = IntPrompt.ask("\n[bold]Number of questions (minimum 1)[/bold]", default=5)
        if question_count >= 1:
            break
        console.print("[red]Question count must be at least 1.[/red]")
    
    console.print("\n[bold]Interview mode:[/bold]")
    console.print("1. Text")
    console.print("2. Voice")
    mode_choice = IntPrompt.ask("Select", choices=["1", "2"])
    interview_mode = "Voice" if mode_choice == 2 else "Text"
    
    return InterviewConfig(
        target_role=target_role,
        experience_level=experience_level,
        interview_type=interview_type,
        question_count=question_count,
        interview_mode=interview_mode
    )

def display_evaluation(evaluation):
    if hasattr(evaluation, "model_dump"):
        evaluation = evaluation.model_dump()

    console.print("\n[bold magenta]════════════════════════════════════════════[/bold magenta]")
    console.print("[bold magenta]             INTERVIEWOS REPORT             [/bold magenta]")
    console.print("[bold magenta]════════════════════════════════════════════[/bold magenta]\n")

    overall_score = evaluation.get('overall_score')
    console.print(Text(f"Overall Score: {overall_score}/100" if overall_score is not None else "Overall Score: Not available (insufficient usable evidence)"))
    console.print(Text(
        "Overall score: mean across usable answers of "
        "(50% correctness/relevance + 30% depth + 20% communication), scaled to 100. "
        "Technical answers use correctness; Behavioral answers use relevance."
    ))
    console.print(Text(f"Evaluation Confidence: {evaluation.get('evaluation_confidence', 'Unknown')}\n"))

    if evaluation.get('technical_score') is not None:
        console.print(Text(f"Technical: {evaluation['technical_score']}/10"))
    if evaluation.get('problem_solving_score') is not None:
        console.print(Text(f"Problem Solving: {evaluation['problem_solving_score']}/10"))
    if evaluation.get('behavioral_score') is not None:
        console.print(Text(f"Behavioral: {evaluation['behavioral_score']}/10"))
    communication_score = evaluation.get('communication_score')
    if communication_score is not None:
        console.print(Text(f"Communication: {communication_score}/10\n"))

    console.print("[bold]Resume Claims:[/bold]")
    for claim in evaluation.get('resume_claims', []):
        console.print(Text(f"- {claim}"))

    console.print("\n[bold]Interview Demonstrated Strengths:[/bold]")
    for strength in evaluation.get('interview_demonstrated_strengths', []):
        console.print(Text(f"✓ {strength}", style="green"))

    console.print("\n[bold]Areas to Improve:[/bold]")
    for weakness in evaluation.get('weaknesses', []):
        console.print(Text(f"! {weakness}", style="yellow"))

    console.print("\n[bold]Role Readiness:[/bold]")
    console.print(Text(str(evaluation.get('role_readiness', 'Not provided'))))
    console.print("\n[bold]Final Feedback:[/bold]")
    console.print(Text(str(evaluation.get('final_feedback', 'Not provided'))))
    console.print("\n[bold magenta]════════════════════════════════════════════[/bold magenta]")


def display_answer_analysis(analysis):
    """Show a candidate score only when the answer supplied usable evidence."""
    classification = analysis.get("answer_classification", "substantive")
    feedback = analysis.get("feedback", "")
    if analysis.get("status") == "failed" or classification == "analysis_failed":
        message = f"Analysis unavailable; no candidate score assigned. {feedback}"
    elif classification == "blank":
        message = f"No answer evidence; no candidate score assigned. {feedback}"
    elif classification == "irrelevant":
        message = f"No relevant answer evidence; no candidate score assigned. {feedback}"
    else:
        message = f"Analysis Score: {analysis.get('correctness_score', 0)}/10 - {feedback}"
    console.print(Text(message, style="dim"))

def review_interview_plan(plan):
    table = Table(title="Interview Plan", show_lines=True)
    table.add_column("#", style="bold cyan", justify="right")
    table.add_column("Type", style="magenta")
    table.add_column("Behavioral category", style="cyan")
    table.add_column("Difficulty", style="yellow")
    table.add_column("Topic", style="blue")
    table.add_column("Question")
    table.add_column("Expected concepts", style="dim")
    for index, question in enumerate(plan, 1):
        table.add_row(
            str(index), question.question_type,
            question.behavioral_category or "—", question.difficulty,
            question.topic, question.question,
            ", ".join(question.expected_concepts),
        )
    console.print(table)

def prepare_interview_plan(profile: CandidateProfile, config: InterviewConfig):
    planner = None
    plan = None
    while plan is None:
        try:
            planner = PlannerService()
            with console.status("[cyan]Preparing the complete interview plan...[/cyan]") as status:
                def show_plan_progress(index, total, category, competency):
                    status.update(
                        f"[cyan]Building interview plan: question {index}/{total} "
                        f"({category}; {competency})...[/cyan]"
                    )

                plan = planner.generate_plan(profile, config, progress_callback=show_plan_progress)
        except Exception as e:
            console.print(f"[red]Could not produce a valid interview plan: {e}[/red]")
            retry = Prompt.ask("Retry plan generation?", choices=["y", "n"], default="y")
            if retry.lower() != "y":
                return None

    while True:
        console.print()
        if config.interview_type == "Mixed" and config.question_count == 1:
            console.print("[yellow]A one-question Mixed interview can only cover Technical. Increase the question count to include Behavioral coverage.[/yellow]")
        for warning in plan.repair_warnings:
            console.print(f"[yellow]Plan repair:[/yellow] {warning}")
        review_interview_plan(plan.questions)
        console.print("[1] Confirm and start  [2] Edit a question  [3] Regenerate one question  [4] Regenerate full plan")
        choice = Prompt.ask("[bold]Plan option[/bold]", choices=["1", "2", "3", "4"], default="1", show_choices=False)
        if choice == "1":
            try:
                planner.validate_plan(plan, config)
            except Exception as e:
                console.print(f"[red]This plan is not valid and cannot be started: {e}[/red]")
                retry = Prompt.ask("Regenerate the full plan?", choices=["y", "n"], default="y")
                if retry.lower() == "y":
                    try:
                        plan = planner.generate_plan(profile, config)
                    except Exception as regenerate_error:
                        console.print(f"[red]Plan regeneration failed: {regenerate_error}[/red]")
                continue
            return plan.questions
        if choice == "2":
            question_number = IntPrompt.ask("Question number to edit", choices=[str(i) for i in range(1, len(plan.questions) + 1)])
            replacement = Prompt.ask("Updated question (leave blank to keep it)", default="")
            if replacement.strip():
                try:
                    validation = planner.edit_question(plan, config, question_number - 1, replacement)
                    console.print(f"[green]Edit accepted:[/green] {validation.reason}")
                except Exception as e:
                    console.print(f"[yellow]Edit rejected; the previous question was kept: {e}[/yellow]")
        elif choice == "3":
            question_number = IntPrompt.ask("Question number to regenerate", choices=[str(i) for i in range(1, len(plan.questions) + 1)])
            try:
                with console.status("[cyan]Regenerating the selected question...[/cyan]"):
                    result = planner.regenerate_question(plan, profile, config, question_number - 1)
                if result.repair_warning:
                    console.print(f"[yellow]Question {question_number} uses a safe fallback:[/yellow] {result.repair_warning}")
            except Exception as e:
                console.print(f"[red]Question regeneration failed; the plan was kept: {e}[/red]")
        else:
            try:
                with console.status("[cyan]Regenerating the full interview plan...[/cyan]"):
                    plan = planner.generate_plan(profile, config)
            except Exception as e:
                console.print(f"[red]Plan regeneration failed; keeping the current plan: {e}[/red]")

def get_voice_answer(audio_service: AudioService, speech_service: SpeechService) -> str:
    while True:
        console.print("\n[bold yellow]🎙 Recording automatically... Speak now, then press ENTER to stop.[/bold yellow]")
        try:
            audio_bytes = audio_service.record_until_keypress()
            console.print("[bold yellow]✓ Recording stopped.[/bold yellow]")
        except AudioServiceError as e:
            console.print(f"[red]Audio capture failed: {e}[/red]")
            return fallback_to_text()

        with console.status("[cyan]⏳ Transcribing...[/cyan]"):
            try:
                transcript = speech_service.transcribe(audio_bytes)
            except SpeechServiceError as e:
                console.print(f"[red]Transcription failed: {e}[/red]")
                return fallback_to_text()
                
        if transcript:
            console.print(f"[bold green]✓ Transcript received:[/bold green] {transcript}")
            return transcript
        else:
            console.print("\n[yellow]I couldn't detect a clear answer.[/yellow]")
            console.print("1. Try again")
            console.print("2. Enter answer manually")
            console.print("3. Skip")
            choice = Prompt.ask("Select", choices=["1", "2", "3"])
            if choice == "1":
                continue
            elif choice == "2":
                return fallback_to_text()
            else:
                return "No answer provided."

def fallback_to_text() -> str:
    console.print("\n[yellow]Falling back to manual text entry.[/yellow]")
    answer = Prompt.ask("\n[bold]Your answer[/bold]")
    return answer

def run_interview_loop(db: Database, initial_state: dict):
    engine = InterviewEngine()
    mode = initial_state['config'].get('interview_mode', 'Text')
    
    audio_service = AudioService() if mode == "Voice" else None
    speech_service = SpeechService() if mode == "Voice" else None
    
    console.print(f"\n[bold green]Starting Interview Engine ({mode} Mode)...[/bold green]\n")
    show_current_question = True
    
    with console.status("[cyan]Generating first question...[/cyan]"):
        state = engine.start_interview(initial_state)
        db.save_session(state)
        
    while state.get("status") == "IN_PROGRESS":
        question_idx = len(state.get("questions", []))
        total_questions = state['config']['question_count']
        remaining_questions = state.get("remaining_question_count", max(0, total_questions - question_idx))
        metadata = state.get("question_metadata", [])
        current_meta = metadata[-1] if metadata else {}
        if current_meta.get("is_follow_up"):
            source_number = (current_meta.get("follow_up_for_plan_index") or 0) + 1
            question_label = f"Follow-up for planned question {source_number}"
        else:
            planned_number = max(1, question_idx - sum(bool(item.get("is_follow_up")) for item in metadata if isinstance(item, dict)))
            question_label = f"Question {planned_number} / {total_questions}"

        if show_current_question:
            console.print(f"\n[bold cyan]{question_label}[/bold cyan] | Planned questions remaining: {remaining_questions} | Difficulty: [yellow]{state['current_difficulty']}[/yellow]")
            console.print(f"Topic: [blue]{state['topics_covered'][-1] if state.get('topics_covered') else 'General'}[/blue]\n")
        
        question_text = state['current_question']
        
        if mode == "Voice" and show_current_question:
            console.print(Panel(question_text, title="Interviewer", border_style="cyan"))
            
            with console.status("[cyan]🔊 Synthesizing speech...[/cyan]"):
                try:
                    audio_bytes = speech_service.speak(question_text)
                    console.print("[cyan]🔊 Speaking question...[/cyan]")
                    audio_service.play_audio(audio_bytes)
                    console.print("[green]✓ Ready[/green]")
                except (SpeechServiceError, AudioServiceError) as e:
                    console.print(f"[red]TTS/Playback failed: {e}[/red]")
                    console.print("[yellow]Continuing with text display only.[/yellow]")
                    
        if mode == "Voice":
            answer = get_voice_answer(audio_service, speech_service)
        else:
            if show_current_question:
                console.print(Panel(question_text, title="Interviewer", border_style="cyan"))
            answer = Prompt.ask("\n[bold]Your answer[/bold]")
            if not answer.strip():
                console.print("[yellow]Please provide an answer.[/yellow]")
                continue
                
        with console.status("[cyan]Analyzing answer and determining next step...[/cyan]"):
            state = engine.process_answer(state, answer)
            db.save_session(state)

        turn_data = state.get("current_turn_intent") or {}
        turn_intent = turn_data.get("intent")
        requested_action = turn_data.get("requested_action")
        if requested_action in {"HINT", "CLARIFY", "REPEAT", "SKIP"}:
            turn_intent = requested_action
        elif (
            requested_action == "EASIER"
            and turn_intent == "ADMITTED_UNKNOWN"
            and state.get("current_question_difficulty", state.get("current_difficulty")) == "Easy"
        ):
            turn_intent = "HINT"
        if turn_intent not in {"ANSWER", "ADMITTED_UNKNOWN", "SKIP"}:
            response = state.get("interviewer_response") or "Would you like to answer, hear a hint, or skip this question?"
            console.print(Panel(response, title="Interviewer", border_style="cyan"))
            if mode == "Voice":
                try:
                    audio_bytes = speech_service.speak(response)
                    audio_service.play_audio(audio_bytes)
                except (SpeechServiceError, AudioServiceError) as error:
                    console.print(f"[yellow]Speech playback unavailable: {error}[/yellow]")
            show_current_question = False
            continue
        if turn_intent == "SKIP":
            console.print("[dim]Question skipped; no candidate score was assigned.[/dim]")
            show_current_question = True
            continue
            
        if state.get('analyses'):
            last_analysis = state['analyses'][-1]
            display_answer_analysis(last_analysis)
        show_current_question = True
            
        if state.get("status") == "COMPLETED":
            console.print("\n[bold green]Interview complete. Generating final report...[/bold green]")
            break

    if state.get("evaluation"):
        if state.get("status") != "COMPLETED":
            console.print("\n[bold green]Interview complete. Displaying the saved evaluation...[/bold green]")
        display_evaluation(state["evaluation"])
    elif state.get("status") == "COMPLETED_WITH_ERRORS":
        console.print("[red]The interview completed, but an error occurred during evaluation.[/red]")
    else:
        console.print("[yellow]The interview ended without a final evaluation.[/yellow]")

def start_interview_setup(db: Database):
    resume_path = get_resume_path()
    resume_service = ResumeService()
    
    with console.status("[bold green]Extracting text...[/bold green]"):
        try:
            text = resume_service.extract_text_from_file(resume_path)
        except ResumeExtractionError as e:
            console.print(f"[red]Error extracting resume: {e}[/red]")
            return
            
    with console.status("[bold green]Analyzing resume with AI...[/bold green]"):
        try:
            profile = resume_service.analyze_resume(text)
        except ResumeExtractionError as e:
            console.print(f"[red]Error analyzing resume: {e}[/red]")
            return
            
    if not profile.skills and not profile.projects and not profile.experience:
        console.print("\n[bold yellow]⚠ Warning:[/bold yellow] The resume contains very little technical information.")
        proceed = Prompt.ask("Continue anyway?", choices=["y", "n"], default="y")
        if proceed.lower() == 'n':
            return
            
    display_profile(profile)
    config = configure_interview(profile)
    try:
        validate_config(config.interview_mode)
    except ValueError as error:
        console.print(f"[red]Configuration Error:[/red] {error}")
        return
    interview_plan = prepare_interview_plan(profile, config)
    if interview_plan is None:
        return
    
    session_id = str(uuid.uuid4())
    state_obj = InterviewState(
        session_id=session_id,
        config=config,
        profile=profile,
        interview_plan=interview_plan,
    )
    
    run_interview_loop(db, state_obj.model_dump(mode="json"))

def main_menu(db: Database):
    while True:
        console.print("\n[bold]Main Menu[/bold]")
        console.print("1. Start Interview")
        console.print("2. Previous Interviews")
        console.print("3. Exit")
        
        choice = Prompt.ask("Select an option", choices=["1", "2", "3"])
        
        if choice == "1":
            start_interview_setup(db)
        elif choice == "2":
            sessions = db.get_all_sessions()
            if not sessions:
                console.print("[yellow]No previous sessions found.[/yellow]")
            else:
                for s in sessions:
                    console.print(f"- {s.session_id} | {s.candidate_name} | {s.target_role} | {s.started_at}")
        elif choice == "3":
            console.print("Goodbye!")
            sys.exit(0)

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Prepare and run an InterviewOS practice interview.")
    return parser

def main(argv=None):
    build_arg_parser().parse_args(argv)
    try:
        validate_config()
    except Exception as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        sys.exit(1)
        
    show_header()
    db = Database()
    main_menu(db)

if __name__ == "__main__":
    main()
