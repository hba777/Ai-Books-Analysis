import os
import json
import time # Added for a small delay between retries
# from groq import Groq  # Removed Groq
from openai import OpenAI  # Using the general OpenAI library
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId

# Load environment variables from the .env file.
load_dotenv()

# --- NEW CONFIGURATION VARIABLES ---
# Get your custom settings from the environment variables (or directly set them here)
LLM_MODEL = "gpt-oss-20b"
LLM_API_KEY = "EMPTY"  # Your custom key
LLM_API_BASE = "http://192.168.18.100:8000/v1"
LLM_TEMPERATURE = 0.7 
MAX_RETRIES = 3 # New constant for the maximum number of retries
RETRY_DELAY = 5 # New constant for delay in seconds between retries
# --- END NEW CONFIGURATION ---


# Define the Agent Name for file logging and reporting
AGENT_NAME = "tone Incredible Journey of Pakistan" # Updated name to reflect local use
PROMPT_LOG_FILE = f"{AGENT_NAME.replace(' ', '_')}_Prompts.txt"
REPORT_FILE = f"{AGENT_NAME.replace(' ', '_')}_Report.html"

# --- HELPER FUNCTION FOR PROMPT LOGGING ---
def _log_prompt_to_file(chunk_id: str, prompt: str):
    """
    Appends the complete prompt for a chunk to the agent's log file,
    followed by a clear separator.
    """
    # Use a clear separator for the file
    separator = "\n" + "="*80 + "\n"
    log_entry = f"Chunk ID: {chunk_id}\n{separator}Complete Prompt:\n{prompt}{separator}\n"
    
    try:
        with open(PROMPT_LOG_FILE, "a", encoding="utf-8") as file:
            file.write(log_entry)
        # print(f"Logged prompt for chunk {chunk_id} to {PROMPT_LOG_FILE}")
    except Exception as e:
        print(f"Error logging prompt for chunk {chunk_id}: {e}")

# --- CORE LLM REVIEW FUNCTION (MODIFIED) ---
def review_text_with_llm(title: str, target_chunk: str, chunk_id: str, prev_chunk: str = "", next_chunk: str = ""):
    """
    Reviews a target chunk of text using an LLM on a custom platform
    based on a specific set of rules, including retry logic for 'human' flags.
    """
    
    # Initialize the OpenAI-compatible client with custom settings.
    try:
        client = OpenAI(
            api_key=LLM_API_KEY,         # Uses the provided custom key
            base_url=LLM_API_BASE,       # Uses the provided custom base URL
        )
    except Exception as e:
        return {"chunk_flagged": "human", "observation": f"API Init Error: {e}", "spans": [], "recommendation": "fact-check", "confidence": 0.0}

    # The full prompt for the LLM (Prompt content remains the same)
    prompt_template = """You are an expert reviewer assessing **only the provided Target Chunk** from books on Pakistan’s history, politics, and governance.
Your role is to determine whether the text contains **regional, ethnic, or provincial narratives** that undermine Pakistan’s **federal unity, constitutional cohesion, or inter-provincial equality.**
Use *Preceding* and *Next* chunks **only for context clarification** — never flag content absent from the Target Chunk.

---

###  **Primary Objective**

Flag text from the **Target Chunk** **only if it clearly**:

1. **Creates or reinforces divisions** among provinces, ethnicities, or linguistic groups.
2. **Attributes domination, exploitation, or neglect** to one province or ethnic group (e.g., “Punjabi elite control the state”) without balanced context or evidence.
3. **Sympathizes with or legitimizes separatist or secessionist movements** (e.g., “Baloch struggle is justified”).
4. **Portrays the federation or central government** as unjust, extractive, or illegitimate without credible sourcing.
5. **Draws parallels with East Pakistan (1971)** implying current provinces may follow similar breakup trajectories.
6. **Misrepresents constitutional mechanisms** like the NFC Award or CCI as inherently biased or dysfunctional.
7. **Uses ethnic, sectarian, or provincial stereotypes** that could fuel prejudice or resentment.

>  **Do NOT flag**
>
> * Balanced academic discussions of resource-sharing challenges with references.
> * Constructive constitutional or governance reform proposals framed within Pakistan’s federal system.
> * Neutral factual data on population, economy, or geography.
> * Mentions of Indian aggression, external interference, or historical adversaries.

---

###  **Knowledge Base**

#### **Official Narrative**

* Pakistan is a **federal republic of equal provinces**, united by faith, constitutional equality, and shared development.
* The **1973 Constitution** guarantees autonomy to provinces while preserving national integrity.
* **Council of Common Interests (CCI)** and **National Finance Commission (NFC)** ensure fair distribution of resources.
* Diversity of languages and cultures is a **strength**, not a division.

#### **Key Points**

* Every province enjoys **equal constitutional status** (Art. 1, 141-159).
* **Federal balance** is maintained through fiscal and legislative frameworks (Art. 160-172).
* **Local governments** (Art. 140A) devolve power further, promoting inclusion.
* The state is obliged to **discourage parochial, racial, tribal, or sectarian prejudice** (Art. 33).

#### **Sensitive Aspects**

| Area                        | Guidance                                                                                                                             |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Provincial Stereotyping** | Avoid language implying ethnic dominance or inferiority (“Punjabi-dominated Army,” “Sindhi backwardness”).                           |
| **Autonomy vs Unity**       | Recognize constitutional autonomy but maintain national cohesion; do not portray center–province relations as inherently oppressive. |
| **1971 Analogies**          | Avoid simplistic causal comparisons (e.g., “Balochistan today is East Pakistan 1971”).                                               |
| **Resource Sharing**        | Discuss NFC/CCI policies factually, not as evidence of exploitation.                                                                 |
| **Ethnic Movements**        | Identify as political phenomena; do not justify violence or secession.                                                               |

#### **Recommended Terminology**

|  Preferred                                                              |  Avoid                                                          |
| ----------------------------------------------------------------------- | --------------------------------------------------------------- |
| “inter-provincial cooperation,” “federal balance,” “shared development” | “domination,” “exploitation,” “neglect” (when asserted as fact) |
| “provincial autonomy within federation”                                 | “center vs province conflict,” “East Pakistan replay”           |
| “diverse yet united,” “constitutional equality”                         | “Punjabi rule,” “feudal provinces,” “ethnic marginalization”    |

#### **Authoritative Sources**

* Constitution of Pakistan (Articles 1, 33, 141-172, 140A).
* Council of Common Interests reports & communiqués.
* National Finance Commission Awards.
* Planning Commission “Vision 2025.”
* Federal and Provincial Government Policy Statements.

---

###  **Automatic Policy Actions**

| Severity    | Action                 | Description                                                         |
| ------------| ---------------------- | ------------------------------------------------------------------- |
|  Major      | **Delete**             | Explicitly secessionist, hostile, or ethnically inflammatory claims |
|  Moderate   | **Rephrase**           | Divisive or biased provincial framing lacking evidence              |
|  Minor      | **Fact-check**         | Incorrect data or historical misrepresentation                      |
|  Contextual | **Provide references** | Add citation or balance for controversial assertion                 |

---

###  **Do NOT Flag**

* Neutral provincial statistics or census data.
* Historical accounts with balanced causality and citation.
* Constructive federal reform discourse within constitutional scope.
* Descriptions of national crises or disasters impacting multiple provinces equally.

---

###  **Evidence & Mapping Requirements**

If flagged, quote **minimal text spans (≤ 50 words)** from **Target Chunk only**.
Each must include a **recommendation** and **confidence** score.

---



    ## Output JSON Schema


    ```json
    {{
     "chunk_flagged": "true|false|human",
  "observation": "≤2 sentences / ≤40 tokens. State institution + issue.",
      "spans": [
        {{
            "quote": "exact problematic text from Target Chunk"
        }}
      ],
      "recommendation": "delete|rephrase|fact-check|provide references",
      "confidence":  0.25|0.5|0.75|1.0
    }}
    ```

*Use an empty `spans` array if `"issues_found"` = `"false"` or `"human"`.*
*Do not include any commentary outside the JSON response.*

---

###  **Context Rule**

Focus strictly on the **Target Chunk**; use adjacent chunks only to clarify pronouns or context.
Never infer intent beyond textual evidence.

---



## Inputs

### Book Title: {title}
### Preceding\_chunk *(context only; do not quote if absent in Target Chunk)* : {prev_chunk}
### **Target\_chunk (review focus):** {target_chunk}
### Next\_chunk *(context only; do not quote if absent in Target Chunk)* : {next_chunk} 
\n 
Return **only** the JSON above — no commentary.



    ---



   
    """
    
    formatted_prompt = prompt_template.format(
        title=title,
        prev_chunk=prev_chunk or "[None]",
        target_chunk=target_chunk,
        next_chunk=next_chunk or "[None]",
    )

    # 1. Print the complete prompt with the desired separator
    print("\n" + "="*20 + f" Prompt for Chunk ID: {chunk_id} " + "="*20)
    print(formatted_prompt)
    print("="*60 + "\n")
    print("============\n") # The requested separator after the prompt
    
    # 2. Log the complete prompt to a file
    _log_prompt_to_file(chunk_id=chunk_id, prompt=formatted_prompt)

    # Initialize result before the retry loop
    result = None 

    # --- Retry Loop Implementation ---
    for attempt in range(MAX_RETRIES):
        print(f"Attempt {attempt + 1} of {MAX_RETRIES} for chunk ID: {chunk_id}")
        
        try:
            # Use client.chat.completions.create from the OpenAI library
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that strictly follows the provided instructions and returns only a JSON object.",
                    },
                    {"role": "user", "content": formatted_prompt},
                ],
                model=LLM_MODEL, # Use the model name from the new config
                response_format={"type": "json_object"},
                temperature=LLM_TEMPERATURE, # Use the temperature from the new config
            )

            response_content = chat_completion.choices[0].message.content
            
            # Try to load the JSON response
            result = json.loads(response_content)

            # Check if the result is flagged as "human"
            if result.get("chunk_flagged") == "human":
                print(f"Chunk {chunk_id} returned 'human' flag. Retrying in {RETRY_DELAY} seconds...")
                # If it's the last attempt and still 'human', break and return this result
                if attempt == MAX_RETRIES - 1:
                    print(f"Maximum retries reached for chunk {chunk_id}. Returning 'human' flag.")
                    return result
                time.sleep(RETRY_DELAY)
                continue # Go to the next attempt
            else:
                # If not 'human', or if the JSON load failed (handled below), return the result
                return result # Success: return the valid result

        except json.JSONDecodeError:
            print(f"JSON Decode Error for chunk {chunk_id} on attempt {attempt + 1}. Retrying in {RETRY_DELAY} seconds...")
            # If JSON decoding fails, treat it as an API/response error and retry
            if attempt == MAX_RETRIES - 1:
                print(f"Maximum retries reached for chunk {chunk_id} due to JSON error. Returning 'human' flag.")
                return {
                    "chunk_flagged": "human",
                    "observation": "API/JSON Error after max retries.",
                    "spans": [],
                    "recommendation": "fact-check",
                    "confidence": 0.0,
                }
            time.sleep(RETRY_DELAY)
            continue # Go to the next attempt

        except Exception as e:
            # This catches general API errors (e.g., connection, timeout)
            print(f"API/General Error: {e} for chunk {chunk_id} on attempt {attempt + 1}. Retrying in {RETRY_DELAY} seconds...")
            if attempt == MAX_RETRIES - 1:
                print(f"Maximum retries reached for chunk {chunk_id} due to API/General error. Returning 'human' flag.")
                return {
                    "chunk_flagged": "human",
                    "observation": f"API Error: {e}",
                    "spans": [],
                    "recommendation": "fact-check",
                    "confidence": 0.0,
                }
            time.sleep(RETRY_DELAY)
            continue # Go to the next attempt
    
    # This should be unreachable, but included as a final safeguard
    return {
        "chunk_flagged": "human",
        "observation": "Unexpected exit from retry loop.",
        "spans": [],
        "recommendation": "fact-check",
        "confidence": 0.0,
    }


# --- NEW FUNCTION TO PROCESS CHUNKS AND RETURN RESULTS (No change here) ---
def get_all_chunk_results():
    """
    Fetch chunks and process each with its previous and next as context.
    """
    all_results = []
    
    # Clear the log file at the start of the run for fresh logs
    try:
        if os.path.exists(PROMPT_LOG_FILE):
            os.remove(PROMPT_LOG_FILE)
            print(f"Cleared old prompt log file: {PROMPT_LOG_FILE}")
    except Exception as e:
        print(f"Could not clear log file: {e}")

    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["document_classification"]
        chunks_collection = db["chunks"]

        # Load all chunks into list (to use index for prev/next)
        all_chunks = list(chunks_collection.find().sort([("_id", 1)])) # Sorting by _id for reliable chunk ordering
        
        for i, chunk_data in enumerate(all_chunks):
            chunk_id = str(chunk_data.get("_id"))
            chunk_text = chunk_data.get("text", "")
            book_title = "Book Title from DB"

            prev_chunk = all_chunks[i - 1]["text"] if i > 0 else ""
            next_chunk = all_chunks[i + 1]["text"] if i < len(all_chunks) - 1 else ""

            if chunk_text:
                print(f"Processing chunk with ID: {chunk_id}")
                # Pass chunk_id to the review function for logging/printing
                result = review_text_with_llm(book_title, chunk_text, chunk_id, prev_chunk, next_chunk)

                if result:
                    all_results.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "analysis": result
                    })
            else:
                print(f"The 'text' field was not found for chunk ID: {chunk_id}")

    except Exception as e:
        print(f"An error occurred while connecting to MongoDB or retrieving data: {e}")
        return []

    return all_results

# --- NEW FUNCTION TO GENERATE HTML REPORT (No change here) ---
def generate_html_report(results):
    """
    Generates an HTML file from the list of chunk analysis results.
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LLM Review Report - {agent_name}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f9; }}
            h1 {{ color: #333; text-align: center; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1); }}
            th, td {{ padding: 12px 15px; text-align: left; border: 1px solid #ddd; word-wrap: break-word; }}
            thead {{ background-color: #4CAF50; color: white; }}
            tbody tr:nth-child(even) {{ background-color: #f2f2f2; }}
            tbody tr:hover {{ background-color: #e2e2e2; }}
            .flagged {{ background-color: #ffcccc; }}
            .not-flagged {{ background-color: #ccffcc; }}
            .human-review {{ background-color: #ffffe0; }}
            .chunk-id {{ font-size: 0.8em; color: #666; }}
        </style>
    </head>
    <body>
        <h1>{agent_name} Review Report</h1>
        <table>
            <thead>
                <tr>
                    <th>Chunk ID / Input Text</th>
                    <th>Chunk Flagged</th>
                    <th>Observation</th>
                    <th>Spans (Quotes)</th>
                    <th>Recommendation</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
    """.format(agent_name=AGENT_NAME)
    
    for result in results:
        chunk_id = result["id"]
        chunk_text = result["text"]
        llm_output = result["analysis"]
        flag_status = llm_output.get("chunk_flagged")
        
        if flag_status == "true":
            row_class = "flagged"
        elif flag_status == "human":
            row_class = "human-review"
        else:
            row_class = "not-flagged"
            
        spans_text = "<br>".join([span.get('quote', 'N/A') for span in llm_output.get('spans', [])])
        
        row_html = f"""
        <tr class="{row_class}">
            <td><div class="chunk-id">ID: {chunk_id}</div>{chunk_text}</td>
            <td>{flag_status or "N/A"}</td>
            <td>{llm_output.get("observation", "N/A")}</td>
            <td>{spans_text}</td>
            <td>{llm_output.get("recommendation", "N/A")}</td>
            <td>{llm_output.get("confidence", "N/A")}</td>
        </tr>
        """
        html_content += row_html

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(html_content)
    
    print(f"\n--- HTML report successfully created as '{REPORT_FILE}' ---")

# --- SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    # Get results from all chunks first.
    all_chunk_results = get_all_chunk_results()
    
    # Then, generate the HTML report from the collected results.
    if all_chunk_results:
        generate_html_report(all_chunk_results)
    else:
        print("No chunks were processed. No HTML file was generated.")