from langgraph.graph import StateGraph, START, END
from app.engine.state import GraphState, normalize_graph_state
from app.engine.nodes import (
    MAX_AMBIGUOUS_TURNS,
    classify_turn_node,
    respond_to_conversation_node,
    prepare_skip_node,
    accept_answer_node,
    generate_question_node,
    analyze_answer_node,
    adaptive_decision_node,
    evaluate_node
)

def route_start(state: GraphState):
    if state.get("status") == "COMPLETED" or state.get("status") == "COMPLETED_WITH_ERRORS":
        return END
    
    # Check if we should finish the interview
    questions_count = len(state.get("questions", []))
    config_count = state['config']['question_count']
    plan = state.get("interview_plan", [])
    plan_order = state.get("plan_execution_order") or list(range(len(plan)))
    plan_exhausted = bool(plan) and state.get("plan_cursor", 0) >= len(plan_order)
    
    if state.get("current_input") is not None:
        return "classify_turn"
    if state.get("current_answer"):
        # We have an answer to analyze
        return "analyze_answer"
    elif plan and state.get("pending_adaptive_question"):
        return "generate_question"
    elif plan_exhausted or (not plan and questions_count >= config_count):
        # Reached the end of planned questions (or legacy no-plan interview).
        return "evaluate"
    else:
        # Need to generate a new question
        return "generate_question"


def route_after_classification(state: GraphState):
    turn = state.get("current_turn_intent") or {}
    intent = turn.get("intent", "AMBIGUOUS")
    requested_action = turn.get("requested_action")
    if requested_action == "SKIP" or intent == "SKIP":
        return "prepare_skip"
    if intent == "AMBIGUOUS" and state.get("consecutive_ambiguous_turns", 0) >= MAX_AMBIGUOUS_TURNS:
        return "prepare_skip"
    if (
        requested_action == "EASIER"
        and intent == "ADMITTED_UNKNOWN"
        and state.get("current_question_difficulty", state.get("current_difficulty")) == "Easy"
    ):
        return "respond_to_conversation"
    if requested_action in {"EASIER", "FOLLOW_UP"} and intent in {"ANSWER", "ADMITTED_UNKNOWN"}:
        return "accept_answer"
    if requested_action in {"HINT", "CLARIFY", "REPEAT"}:
        return "respond_to_conversation"
    if intent in {"ANSWER", "ADMITTED_UNKNOWN"}:
        return "accept_answer"
    return "respond_to_conversation"

def route_after_decision(state: GraphState):
    questions_count = len(state.get("questions", []))
    config_count = state['config']['question_count']
    
    # Check if the AI decided to end early or if we reached the count
    last_decision = state['decisions'][-1]['action'] if state.get("decisions") else None
    plan = state.get("interview_plan", [])
    plan_order = state.get("plan_execution_order") or list(range(len(plan)))
    plan_exhausted = bool(plan) and state.get("plan_cursor", 0) >= len(plan_order)
    pending_follow_up = bool((state.get("pending_adaptive_question") or {}).get("is_follow_up"))

    if pending_follow_up:
        return "generate_question"
    if last_decision == "FINISH" and (not plan or plan_exhausted):
        return "evaluate"
    if plan and plan_exhausted:
        return "evaluate"
    if not plan and (last_decision == "FINISH" or questions_count >= config_count):
        return "evaluate"
    else:
        return "generate_question"

def build_interview_graph():
    builder = StateGraph(GraphState)
    
    # Add nodes
    builder.add_node("classify_turn", classify_turn_node)
    builder.add_node("respond_to_conversation", respond_to_conversation_node)
    builder.add_node("prepare_skip", prepare_skip_node)
    builder.add_node("accept_answer", accept_answer_node)
    builder.add_node("generate_question", generate_question_node)
    builder.add_node("analyze_answer", analyze_answer_node)
    builder.add_node("adaptive_decision", adaptive_decision_node)
    builder.add_node("evaluate", evaluate_node)
    
    # Add edges
    builder.add_conditional_edges(START, route_start)
    builder.add_conditional_edges("classify_turn", route_after_classification)
    builder.add_edge("respond_to_conversation", END)
    builder.add_edge("prepare_skip", "analyze_answer")
    builder.add_edge("accept_answer", "analyze_answer")
    builder.add_edge("generate_question", END)  # Stop after asking to wait for user input
    builder.add_edge("analyze_answer", "adaptive_decision")
    builder.add_conditional_edges("adaptive_decision", route_after_decision)
    builder.add_edge("evaluate", END)
    
    return builder.compile()

class InterviewEngine:
    def __init__(self):
        self.graph = build_interview_graph()
        
    def start_interview(self, initial_state: GraphState) -> GraphState:
        # Run graph from START. It will route to generate_question and then END.
        normalized = normalize_graph_state(initial_state)
        return normalize_graph_state(self.graph.invoke(normalized))
        
    def process_answer(self, state: GraphState, answer: str) -> GraphState:
        normalized = normalize_graph_state(state)
        normalized["current_input"] = answer
        normalized["current_answer"] = None
        normalized["interviewer_response"] = None
        # Classify every new utterance before deciding whether it enters the
        # existing answer analysis workflow.
        return normalize_graph_state(self.graph.invoke(normalized))
