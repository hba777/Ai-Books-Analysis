import json
import pymongo
import re
from typing import List, Dict, Callable
from langgraph.graph import END, StateGraph
from langchain_core.runnables import RunnableLambda
from models import State
from knowledge_base import get_relevant_info
from config import MONGO_URI, AGENTS_DB_NAME, AGENTS_COLLECTION_NAME
from generate_prompt import build_prompt
from datetime import datetime

# Define type for agent functions
Agent = Callable[[State], Dict]

# Dictionary to hold all agents
available_agents: Dict[str, Agent] = {}

# --- New Functions for Text Preprocessing ---
def split_chunk_into_lines(text):
    """
    Splits a long chunk into multiple logical lines for better Groq parsing.
    Splits first on periods, then on commas if the line is too long.
    """
    sentences = re.split(r'(?<=[.]) +', text)
    lines = []
    for s in sentences:
        if len(s) > 120:
            lines.extend([x.strip() for x in s.split(",") if x.strip()])
        else:
            lines.append(s.strip())
    return "\n".join(lines)

def format_long_text_as_target_chunk(long_text):
    """
    Converts any long multi-paragraph text into a single-line target_chunk
    with references preserved.
    """
    text = re.sub(r'\s+', ' ', long_text).strip()
    text = re.sub(r'(\d{3} )', r'; \1', text)
    return text
# --- End of New Functions ---

def parse_and_validate_output(raw_output: str) -> dict:
    """
    Parses raw LLM output, handles errors, and ensures all required keys are present.
    """
    default_output = {
        "chunk_flagged": "human",
        "observation": "Failed to parse LLM output. Requires human review.",
        "spans": [],
        "recommendation": "fact-check",
        "confidence": 0.0
    }
    
    # 1. Try to extract a JSON block (e.g., from ```json...```)
    try:
        match = re.search(r'```json(.*?)```', raw_output, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            parsed_data = json.loads(json_str)
        else:
            # If no JSON block is found, try to parse the entire string
            parsed_data = json.loads(raw_output)

        # 2. Check for required keys and provide defaults if missing
        for key, default_value in default_output.items():
            if key not in parsed_data:
                print(f"⚠️ Warning: Missing key '{key}' in parsed data. Providing default value.")
                parsed_data[key] = default_value

        return parsed_data
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"❌ JSON parsing error: {e}. Raw output: '{raw_output}'")
        return default_output


def register_agent(name: str, agent_function: Agent):
    """Register an agent function."""
    available_agents[name] = agent_function

def create_review_agent(review_name: str, confidence_score: int, llm_model, eval_llm_model) -> Agent:
    """
    Creates a specialized review agent function that includes an internal evaluation loop.
    """
    def agent_sub_step(state: State) -> State:
        print(f"\n--- {state['current_agent_name']} Sub-Agent Step - Attempt {state.get('current_agent_retries', 0) + 1} ---")
        report_text = state["report_text"]
        metadata = state["metadata"]
        
        target_chunk = format_long_text_as_target_chunk(report_text)
        formatted_chunk = split_chunk_into_lines(target_chunk)
        
        # Get the previous and next chunks from the state's metadata
        previous_chunk = metadata.get("previous_chunk", "")
        next_chunk = metadata.get("next_chunk", "")

        prompt = build_prompt(
            agent_name=state["current_agent_name"],
            title=metadata.get("title", "N/A"),
            target_chunk=formatted_chunk,
            previous_chunk=previous_chunk,
            next_chunk=next_chunk
        )

        print(f"--- {state['current_agent_name']} Generated Prompt ---")
        print(prompt)
        print("-" * 30)

        response = llm_model.invoke(prompt)
        raw_output = response.content

        # Use the new, more robust parsing function
        parsed_output = parse_and_validate_output(raw_output)

        return {
            "current_agent_input_prompt": prompt,
            "current_agent_raw_output": raw_output,
            "current_agent_parsed_output": parsed_output,
            "current_agent_retries": state.get("current_agent_retries", 0) + 1,
        }

    def evaluation_sub_step(state: State) -> State:
        # If the parsed output indicates a parsing failure, skip evaluation and set human review flag
        parsed_output = state.get("current_agent_parsed_output", {})
        if parsed_output.get("chunk_flagged") == "human" and "Failed to parse" in parsed_output.get("observation", ""):
            print(f"\n--- {state['current_agent_name']} Evaluation Skipped due to JSON Decode Error ---")
            return {"current_agent_confidence": 0, "current_agent_human_review": True}

        prompt_from_agent = state["current_agent_input_prompt"]
        response_from_agent = state["current_agent_raw_output"]

        eval_prompt = f"""
Evaluate the following:

Prompt given to agent:
{prompt_from_agent}

Agent's Raw Response:
"{response_from_agent}"

How correct and relevant is the Response to the Prompt?

Give a confidence score between 0 and 100.

Respond only with a single, valid JSON object that follows this structure:
{{"confidence": <score>}}
DO NOT include any explanation or text outside of the JSON object.
"""

        eval_response = eval_llm_model.invoke(eval_prompt).content
        confidence = 0
        try:
            eval_data = json.loads(eval_response)
            confidence = int(eval_data.get("confidence", 0))
        except json.JSONDecodeError:
            confidence = 0

        return {"current_agent_confidence": confidence, "current_agent_human_review": False}


    def route_sub_step(state: State) -> str:
        max_retries = 3
        parsed_output = state.get("current_agent_parsed_output", {})
        current_agent_retries = state.get("current_agent_retries", 0)
        current_agent_confidence = state.get("current_agent_confidence", 0)
        current_agent_human_review = state.get("current_agent_human_review", False)

        # 1. First, check if the parsed output itself indicates a human review is needed (e.g., due to a parsing error)
        if parsed_output.get("chunk_flagged") == "human" or current_agent_human_review:
            print("❗ Routing to human review due to parsing failure or LLM's own 'human' flag.")
            return "human_review_needed_sub_step"

        # 2. Then, check if the confidence score is too low after all retries
        if current_agent_confidence < confidence_score:
            print(f"⚠️ Confidence score ({current_agent_confidence}%) is too low.")
            if current_agent_retries >= max_retries:
                print("❗ Max retries exceeded. Routing to human review.")
                return "human_review_needed_sub_step"
            else:
                print("🔄 Retrying agent step.")
                return "agent_sub_step"

        # 3. If confidence is high enough, the process is complete
        print("✅ Confidence score is sufficient. Ending agent sub-workflow.")
        return "end"

    def human_review_sub_step(state: State) -> State:
        return {"current_agent_human_review": True}

    # Build agent graph
    agent_graph_builder = StateGraph(State)
    agent_graph_builder.add_node("agent_sub_step", RunnableLambda(agent_sub_step))
    agent_graph_builder.add_node("evaluation_sub_step", RunnableLambda(evaluation_sub_step))
    agent_graph_builder.add_node("human_review_needed_sub_step", RunnableLambda(human_review_sub_step))

    agent_graph_builder.set_entry_point("agent_sub_step")
    agent_graph_builder.add_edge("agent_sub_step", "evaluation_sub_step")
    agent_graph_builder.add_conditional_edges(
        "evaluation_sub_step",
        route_sub_step,
        {
            "agent_sub_step": "agent_sub_step",
            "human_review_needed_sub_step": "human_review_needed_sub_step",
            "end": END
        }
    )
    agent_graph_builder.add_edge("human_review_needed_sub_step", END)
    agent_sub_graph = agent_graph_builder.compile()

    def review_agent_with_evaluation(state: State) -> Dict:
        initial_sub_state = {
            "report_text": state["report_text"],
            "metadata": state["metadata"],
            "current_agent_name": review_name,
            "current_agent_retries": 0,
            "current_agent_confidence": 0,
            "current_agent_human_review": False,
            "final_decision_report": "",
            "aggregate": [],
            "main_node_output": {}
        }

        final_sub_state = agent_sub_graph.invoke(initial_sub_state)

        agent_result = final_sub_state.get("current_agent_parsed_output", {"error": "No output parsed"})
        agent_confidence = final_sub_state.get("current_agent_confidence", 0)
        agent_retries = final_sub_state.get("current_agent_retries", 0)
        agent_human_review = final_sub_state.get("current_agent_human_review", False)

        return {
            review_name: agent_result,
            "aggregate": [f"{review_name} Output: {agent_result} (Confidence: {agent_confidence}%, Retries: {agent_retries}, Human Review: {agent_human_review})"],
            "main_node_output": {
                review_name: {
                    "output": agent_result,
                    "confidence": agent_confidence,
                    "retries": agent_retries,
                    "human_review": agent_human_review
                }
            }
        }

    return review_agent_with_evaluation


def load_agents_from_mongo(llm_model, eval_llm_model):
    """Load all agents from MongoDB and register only those with type='analysis'."""
    client = None
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[AGENTS_DB_NAME]
        collection = db[AGENTS_COLLECTION_NAME]

        # Only fetch agents where type is 'analysis'
        rows = collection.find({"type": "analysis"})

        for doc in rows:
            agent_name = doc.get("agent_name")
            confidence_score = doc.get("confidence_score", 0)
            agent_type = doc.get("type")

            if agent_type != "analysis":
                print(f"⏩ Skipping agent '{agent_name}' (type={agent_type})")
                continue

            if agent_name and confidence_score is not None:
                agent = create_review_agent(agent_name, confidence_score, llm_model, eval_llm_model)
                register_agent(agent_name, agent)
                print(f"✅ Agent '{agent_name}' (type={agent_type}) loaded with confidence score: {confidence_score}")
            else:
                print(f"⚠️ Error: Missing 'agent_name' or 'confidence_score' in document: {doc}")

    except pymongo.errors.ConnectionFailure as e:
        print(f"MongoDB connection error: {e}")
    finally:
        if client:
            client.close()