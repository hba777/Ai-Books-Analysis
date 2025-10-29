from typing_extensions import TypedDict, Annotated, List, Dict
import operator

# --- STATE DEFINITION ---
class State(TypedDict):
    """
    Represents the state of the Langgraph workflow.

    Attributes:
        report_text (str): The main text segment being analyzed.
        final_decision_report (str): The compiled final report of all reviews.
        aggregate: Annotated[List[str], operator.add]): A list to accumulate
                                     messages/outputs from agents.
        main_node_output (Annotated[Dict, lambda a, b: {**a, **b}]): Output
                                       from the main node.
        metadata (Dict): Contains contextual information about the report_text
                        (e.g., page, paragraph, title).
        # New fields for per-agent evaluation
        current_agent_name: str # To keep track of which agent is running its sub-workflow
        current_agent_input_prompt: str # The exact prompt sent to the agent's LLM
        current_agent_raw_output: str # The raw string output from the agent's LLM
        current_agent_parsed_output: Dict # The parsed JSON output from the agent
        current_agent_confidence: int
        current_agent_retries: int
        current_agent_human_review: bool
    """
    report_text: str
    final_decision_report: str
    aggregate: Annotated[List[str], operator.add]
    main_node_output: Annotated[Dict, lambda a, b: {**a, **b}]
    metadata: Dict
    current_agent_name: str
    current_agent_input_prompt: str
    current_agent_raw_output: str
    current_agent_parsed_output: Dict
    current_agent_confidence: int
    current_agent_retries: int
    current_agent_human_review: bool