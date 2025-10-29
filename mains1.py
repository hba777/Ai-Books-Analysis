from langgraph.graph import START, END, StateGraph
from models import State
from llm_init import llm, eval_llm, llm1
from knowledge_base import knowledge_list, retriever
from agents import load_agents_from_mongo, available_agents
from workflow_nodes import main_node, final_report_generator
# Modified imports to use Pipeline 1 specific chunk retrieval functions
# Now importing the new functions from pdf_processor
from pdf_processor import get_first_pipeline1_chunk, get_all_pipeline1_chunks_details, get_next_pending_pipeline1_chunk, get_all_pending_pipeline1_chunks_details, get_chunk_with_context
from config import AGENTS_DB_NAME, AGENTS_COLLECTION_NAME, MONGO_URI, PDF_DB_NAME
from database_saver import save_results_to_mongo, clear_results_collection, update_chunk_analysis_status, RESULTS_DB_NAME, RESULTS_COLLECTION_NAME
from text_classifier import classify_text
import pymongo
import json
from datetime import datetime

def run_workflow():
    # Clear the results collection at the beginning of each program execution
    # Consider if you really want to clear all results every time you run.
    # If you're resuming, you might not want to clear previous results.
    # clear_results_collection() # <--- COMMENTED OUT TO PRESERVE PREVIOUS RUNS' DATA

    # Load agents dynamically from MongoDB
    print("Loading agents from MongoDB...")
    load_agents_from_mongo(llm, eval_llm)
    
    total_agents = len(available_agents)
    if total_agents == 0:
        print("WARNING: No agents loaded. Analysis workflow might not function as expected.")
        return # Exit if no agents are loaded

    # Initialize the StateGraph with the defined State
    graph_builder = StateGraph(State)

    # Add the main_node and final_report_generator nodes to the graph
    graph_builder.add_node("main_node", main_node)
    graph_builder.add_node("fnl_rprt", final_report_generator)

    # Add dynamically loaded agents as nodes
    for agent_name, agent_runnable in available_agents.items():
        graph_builder.add_node(agent_name, agent_runnable)
        print(f"Added agent '{agent_name}' as a node to the graph.")

    # Set the entry point of the graph to "main_node"
    graph_builder.add_edge(START, "main_node")

    # Dynamically add edges from "main_node" to each loaded agent, and then from each agent to the "fnl_rprt"
    for agent_name in available_agents:
        graph_builder.add_edge("main_node", agent_name)
        graph_builder.add_edge(agent_name, "fnl_rprt")

    # Set the exit point of the graph to "fnl_rprt"
    graph_builder.add_edge("fnl_rprt", END)

    # Compile the graph for execution
    graph = graph_builder.compile()

    print("Loading chunks from Pipeline 1's database...")

    # --- CHOOSE ONE OPTION BELOW: Process next pending chunk OR Process all pending chunks ---

    # OPTION 1: Process only the NEXT pending Pipeline 1 chunk
    # Uncomment the following two lines and comment out OPTION 2 to use this.
    #print("\n--- OPTION 1: Executing graph.invoke() for the next PENDING chunk from Pipeline 1 ---")
    #next_pending_chunk = get_next_pending_pipeline1_chunk()
    #documents_to_process = [next_pending_chunk] if next_pending_chunk else []

    # OPTION 2: Process ALL PENDING Pipeline 1 chunks (ACTIVE BY DEFAULT)
    # Uncomment the following two lines and comment out OPTION 1 to use this.
    print("\n--- OPTION 2: Executing graph.invoke() for ALL PENDING chunks from Pipeline 1 ---")
    documents_to_process = get_all_pending_pipeline1_chunks_details()


    # --- Common processing loop for selected documents (Pipeline 1 schema) ---
    if documents_to_process:
        print(f"Found {len(documents_to_process)} PENDING chunks from Pipeline 1 to process.")
        for doc_to_process in documents_to_process:
            if not doc_to_process:
                continue

            # Fetch the target chunk along with its surrounding context
            previous_chunk, target_chunk, next_chunk = get_chunk_with_context(
                doc_id=doc_to_process.get("doc_id"),
                chunk_index=doc_to_process.get("chunk_index")
            )
            
            # Extract fields from the target chunk
            p1_chunk_uuid = target_chunk.get("_id")
            doc_id_p1 = target_chunk.get("doc_id")
            chunk_index_p1 = target_chunk.get("chunk_index")
            original_chunk_text = target_chunk.get("text")
            book_name_p1 = target_chunk.get("doc_name", "Unknown Document")
            p1_coordinates = target_chunk.get("coordinates")
            p1_page_number = target_chunk.get("page_number")
            
            # Get text for previous and next chunks
            previous_chunk_text = previous_chunk.get("text", "") if previous_chunk else ""
            next_chunk_text = next_chunk.get("text", "") if next_chunk else ""

            # Use the target chunk's text for classification
            merged_text_for_id = original_chunk_text

            print(f"\n--- Processing Chunk ID: {p1_chunk_uuid} (Document: '{book_name_p1}', P1 Doc ID: {doc_id_p1}, P1 Chunk Index: {chunk_index_p1}) ---")
            print(f"Original Chunk Text: {original_chunk_text}\n")

            classification_result = classify_text(merged_text_for_id)
            predicted_label = classification_result['predicted_label']
            print(f"--- Predicted Label for Chunk: \"{predicted_label}\" (Confidence: {classification_result['confidence']}%) ---")

            report_data = {
                "report_text": merged_text_for_id,
                "metadata": {
                    "doc_id": doc_id_p1,
                    "chunk_index": chunk_index_p1,
                    "title": book_name_p1,
                    "chunk_id": p1_chunk_uuid,
                    "predicted_label": predicted_label,
                    "classification_scores": classification_result['all_scores'],
                    "coordinates": p1_coordinates,
                    "page_number": p1_page_number,
                    "previous_chunk": previous_chunk_text,
                    "next_chunk": next_chunk_text,
                },
                "main_node_output": {},
                "aggregate": [],
                "final_decision_report": "",
                "current_agent_name": "",
                "current_agent_input_prompt": "",
                "current_agent_raw_output": "",
                "current_agent_parsed_output": {},
                "current_agent_confidence": 0,
                "current_agent_retries": 0,
                "current_agent_human_review": False
            }

            print(f"\n--- Langgraph Workflow Input for Chunk ID: {p1_chunk_uuid} ---")
            print("Initial state before agent execution. Individual agents will now perform their internal evaluation loops.")
            print("-" * 40)

            result_with_review = graph.invoke(report_data)

            overall_chunk_status = "Complete"
            # Initialize agent_analysis_statuses with all available agents set to "Pending"
            agent_analysis_statuses = {agent_name: "Pending" for agent_name in available_agents.keys()}
            
            # Now iterate over agents that actually produced output and update their status
            for agent_name, agent_data in result_with_review.get("main_node_output", {}).items():
                agent_output = agent_data.get("output", {})

                # --- MODIFIED LOGIC: First, check for the specific `None` case as per your request ---
                # This ensures that if the agent returns None for the key fields, the status is 'Complete'.
                if (
                    agent_output.get("problematic_text") is None
                    and agent_output.get("observation") is None
                    and agent_output.get("recommendation") is None
                ):
                    agent_analysis_statuses[agent_name] = "Complete"
                else:
                    # --- EXISTING LOGIC: If it's not the `None` case, run the original validation ---
                    is_output_complete = True
                    
                    if not isinstance(agent_output, dict):
                        is_output_complete = False
                    else:
                        if "issues_found" in agent_output and not isinstance(agent_output.get("issues_found"), bool):
                            is_output_complete = False
                        
                        if "observation" in agent_output and not isinstance(agent_output.get("observation"), str):
                            is_output_complete = False
                        
                        if "recommendation" in agent_output and not isinstance(agent_output.get("recommendation"), str):
                            is_output_complete = False

                    if is_output_complete:
                        agent_analysis_statuses[agent_name] = "Complete"
                    else:
                        agent_analysis_statuses[agent_name] = "Pending"
                        overall_chunk_status = "Pending" # If any agent is pending, the overall chunk is pending

            save_results_to_mongo(
                chunk_uuid=p1_chunk_uuid,
                doc_id=doc_id_p1,
                chunk_index=chunk_index_p1,
                report_text=original_chunk_text,
                book_name=book_name_p1,
                predicted_label=predicted_label,
                classification_scores=classification_result['all_scores'],
                coordinates=p1_coordinates,
                page_number=p1_page_number,
                result_with_review=result_with_review,
                overall_chunk_status=overall_chunk_status,
                agent_analysis_statuses=agent_analysis_statuses
            )
            
            update_chunk_analysis_status(
                doc_id=doc_id_p1,
                chunk_id=p1_chunk_uuid,
                analysis_status=overall_chunk_status
            )

            print("\n--- Langgraph Workflow Final Output (from State) ---")
            for agent_name, agent_output_data in result_with_review.get("main_node_output", {}).items():
                print(f"\n--- Summary for {agent_name} ---\n")
                output_content = agent_output_data.get('output', {})
                # Updated lines to print the new JSON fields
                print(f"  Chunk Flagged: {output_content.get('chunk_flagged', 'N/A')}")
                print(f"  Observation: {output_content.get('observation', 'N/A')}")
                print(f"  Recommendation: {output_content.get('recommendation', 'N/A')}")
                # These lines were already correct, but moved here for clarity
                print(f"  Confidence: {agent_output_data.get('confidence', 0)}%")
                print(f"  Retries: {agent_output_data.get('retries', 0)}")
                print(f"  Human Review Needed: {agent_output_data.get('human_review', False)}")
            
            print(f"\n--- Overall Chunk Status: {overall_chunk_status} ---\n")
            print(f"--- Agent Analysis Statuses (per chunk, all agents included): {agent_analysis_statuses} ---\n")

            print("Full Result Dictionary (for debugging):\n")
            print(result_with_review)
            print("-" * 40)

    else:
        print("No PENDING chunks found from Pipeline 1's configured database and collection to process. All chunks might be processed, or none were pending.")

if __name__ == "__main__":
    run_workflow()