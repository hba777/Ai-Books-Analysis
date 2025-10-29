from langgraph.graph import START, END, StateGraph
from models import State
from llm_init import llm, eval_llm, llm1
from knowledge_base import knowledge_list, retriever
from agents import load_agents_from_mongo, available_agents
from workflow_nodes import main_node, final_report_generator
from pdf_processor import get_first_pipeline1_chunk, get_all_pipeline1_chunks_details, get_next_pending_pipeline1_chunk, get_all_pending_pipeline1_chunks_details
from config import AGENTS_DB_NAME, AGENTS_COLLECTION_NAME
from database_saver import save_results_to_mongo, clear_results_collection, update_chunk_analysis_status
from text_classifier import classify_text
import fitz
import json

def find_text_coordinates_in_pdf(pdf_path: str, search_text: str):
    """
    Searches for a text string in a PDF, handling multi-line and single-line cases.
    Returns a list of dictionaries, each containing page number and bbox.
    """
    if not search_text:
        return []

    cleaned_search_text = " ".join(search_text.split())
    search_words = cleaned_search_text.split()
    
    coordinates_list = []
    
    try:
        doc = fitz.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Use `get_text("words")` to get a list of all words and their bboxes on the page
            words_on_page = page.get_text("words")
            
            # Create a list of tuples: (word, bbox)
            word_bboxes = [(word[4], fitz.Rect(word[:4])) for word in words_on_page]
            
            # Simple word-based matching
            for i in range(len(word_bboxes) - len(search_words) + 1):
                # Check if the sequence of words matches
                match = True
                for j in range(len(search_words)):
                    if search_words[j] != word_bboxes[i+j][0]:
                        match = False
                        break
                
                if match:
                    # Found a match, merge the bboxes
                    combined_bbox = word_bboxes[i][1]
                    for j in range(1, len(search_words)):
                        combined_bbox.include_rect(word_bboxes[i+j][1])
                    
                    coordinates_list.append({
                        "page": page_num + 1,
                        "bbox": [combined_bbox.x0, combined_bbox.y0, combined_bbox.x1, combined_bbox.y1]
                    })
            
        doc.close()
    except Exception as e:
        print(f"Error finding coordinates in PDF: {e}")
        return []
    
    return coordinates_list

def run_workflow(pdf_path):
    print("Loading agents from MongoDB...")
    load_agents_from_mongo(llm, eval_llm)
    
    total_agents = len(available_agents)
    if total_agents == 0:
        print("WARNING: No agents loaded. Analysis workflow might not function as expected.")
        return

    graph_builder = StateGraph(State)

    graph_builder.add_node("main_node", main_node)
    graph_builder.add_node("fnl_rprt", final_report_generator)

    for agent_name, agent_runnable in available_agents.items():
        graph_builder.add_node(agent_name, agent_runnable)
        print(f"Added agent '{agent_name}' as a node to the graph.")

    graph_builder.add_edge(START, "main_node")

    for agent_name in available_agents:
        graph_builder.add_edge("main_node", agent_name)
        graph_builder.add_edge(agent_name, "fnl_rprt")

    graph_builder.add_edge("fnl_rprt", END)

    graph = graph_builder.compile()

    print("Loading chunks from Pipeline 1's database...")

    print("\n--- OPTION 2: Executing graph.invoke() for ALL PENDING chunks from Pipeline 1 ---")
    documents_to_process = get_all_pending_pipeline1_chunks_details()


    if documents_to_process:
        print(f"Found {len(documents_to_process)} PENDING chunks from Pipeline 1 to process.")
        for doc_to_process in documents_to_process:
            if not doc_to_process:
                continue

            p1_chunk_uuid = doc_to_process.get("chunk_id")
            doc_id_p1 = doc_to_process.get("doc_id")
            chunk_index_p1 = doc_to_process.get("chunk_index")
            original_chunk_text = doc_to_process.get("text")
            book_name_p1 = doc_to_process.get("doc_name", "Unknown Document")

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
                    "classification_scores": classification_result['all_scores']
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
            agent_analysis_statuses = {agent_name: "Pending" for agent_name in available_agents.keys()}
            
            for agent_name, agent_data in result_with_review.get("main_node_output", {}).items():
                agent_output = agent_data.get("output", {})
                
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
                    problematic_text = agent_output.get("problematic_text")
                    if problematic_text and problematic_text.lower() != 'n/a':
                        coordinates = find_text_coordinates_in_pdf(pdf_path, problematic_text)
                        agent_output["problematic_text_coordinates"] = coordinates
                        print(f"✅ Coordinates for {agent_name} added: {coordinates}")
                    else:
                        agent_output["problematic_text_coordinates"] = []
                        
                else:
                    agent_analysis_statuses[agent_name] = "Pending"
                    overall_chunk_status = "Pending"

            # --- MODIFICATION: Convert the state object to a dictionary for MongoDB compatibility ---
            results_to_save = dict(result_with_review)
            # --- MODIFICATION END ---
            
            save_results_to_mongo(
                chunk_uuid=p1_chunk_uuid,
                doc_id=doc_id_p1,
                chunk_index=chunk_index_p1,
                report_text=original_chunk_text,
                book_name=book_name_p1,
                predicted_label=predicted_label,
                classification_scores=classification_result['all_scores'],
                result_with_review=results_to_save,
                overall_chunk_status=overall_chunk_status,
                agent_analysis_statuses=agent_analysis_statuses
            )
            
            update_chunk_analysis_status(
                doc_id=doc_id_p1,
                chunk_id=p1_chunk_uuid,
                analysis_status=overall_chunk_status
            )

            print("\n--- Langgraph Workflow Final Output (from State) ---")
            for agent_name, agent_output_data in result_with_review["main_node_output"].items():
                print(f"\n--- Summary for {agent_name} ---\n")
                output_content = agent_output_data.get('output', {})
                print(f"  Parsed Output: {output_content.get('problematic_text', 'No problematic text found.')}")
                print(f"  Coordinates: {output_content.get('problematic_text_coordinates', 'N/A')}")
                print(f"  Observation: {output_content.get('observation', 'N/A')}")
                print(f"  Recommendation: {output_content.get('recommendation', 'N/A')}")
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
    pdf_file_path = "The Lost of War.pdf"
    run_workflow(pdf_file_path)