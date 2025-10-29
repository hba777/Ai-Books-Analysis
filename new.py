import os
import json
import time
import re
from openai import OpenAI
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId
from typing import Dict, Any, List

# Load environment variables from the .env file.
# Make sure your .env file is in the same directory.
load_dotenv()

# --- HARDCODED CONFIGURATION VARIABLES ---
# NOTE: LLM_API_KEY is not read from .env in this configuration, 
# it is explicitly set to "EMPTY" as per the original code.
LLM_MODEL = "gpt-oss-20b"
LLM_API_KEY = "EMPTY" 
LLM_API_BASE = "http://192.168.18.100:8000/v1"
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB_NAME = "document_classification"
MONGO_CHUNK_COLLECTION = "chunks"
# --- END HARDCODED CONFIGURATION ---

MAX_RETRIES = 3
RETRY_DELAY = 5

# Define the Agent Name for file logging and reporting
AGENT_NAME = "National_Security_Agent full book_new"
PROMPT_LOG_FILE = f"{AGENT_NAME.replace(' ', '_')}_Prompts.txt"
# NEW: Define a separate log file for Evaluation Prompts
EVAL_PROMPT_LOG_FILE = f"{AGENT_NAME.replace(' ', '_')}_Evaluation_Prompts.txt"
REPORT_FILE = f"{AGENT_NAME.replace(' ', '_')}_Report.html"

# --- HELPER FUNCTION FOR PROMPT LOGGING (UNICODE FIX APPLIED) ---
def _log_prompt_to_file(chunk_id: str, prompt: str, log_filename: str):
    """
    Appends the complete prompt for a chunk to the specified log file, ensuring 
    UTF-8 encoding for Unicode compatibility. This function is now generic 
    for both main and evaluation prompts.
    """
    separator = "\n" + "="*80 + "\n"
    log_entry = f"Chunk ID: {chunk_id}\n{separator}Complete Prompt:\n{prompt}{separator}\n"
    
    try:
        # Explicitly setting encoding="utf-8" for robust file writing
        with open(log_filename, "a", encoding="utf-8") as file:
            file.write(log_entry)
    except Exception as e:
        print(f"Error logging prompt for chunk {chunk_id} to {log_filename}: {e}")

# --- EVALUATION FUNCTION (FOR LLM RESPONSE QUALITY) ---
# chunk_id is now passed to this function for logging purposes
def evaluate_agent_response(chunk_id: str, agent_prompt: str, agent_response: str) -> Dict[str, Any]:
    """
    Sends the agent's prompt and raw response to the LLM for quality evaluation.
    Includes 3 retry attempts, robust JSON cleanup, partial recovery, and safe fallbacks.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 3  # seconds between attempts

    try:
        eval_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE)
    except Exception as e:
        print(f"API Init Error for evaluator: {e}")
        return {"confidence": 50}  # fallback

    # --- Evaluation Prompt ---
    eval_prompt_template = f"""
You are an impartial evaluator reviewing the quality of an AI Agent’s response to a policy-based text-screening task.

### Task Context
Each Agent receives a prompt describing Pakistan’s official narratives and a book “Target Chunk” to review.  
The Agent must return a JSON output that:
1. **Flags or clears the chunk** correctly based on the policy rules in its prompt.
2. Provides **accurate, policy-aligned reasoning** — not hallucinated, off-topic.

Your job is to assess **how correct and relevant** the Agent’s response is with respect to the given prompt.

---

### Evaluation Criteria
1. **Task Fulfillment**
2. **Relevance & Accuracy**
3. **Format & Compliance**

---

### Scoring Scale (30–100)
| Range | Meaning |
|-------|----------|
| **31–60** | Partially relevant |
| **61–85** | Generally correct, minor lapses |
| **86–100** | Fully correct, policy-aligned |

---

### 🧾 Inputs
Prompt given to Agent:
{agent_prompt}

Agent's Raw Response:
{agent_response}

---

### Output Format
Respond **only** with a valid JSON:
{{"confidence": <integer between 30 and 100>}}
"""
    # Log the prompt
    _log_prompt_to_file(chunk_id=chunk_id, prompt=eval_prompt_template, log_filename=EVAL_PROMPT_LOG_FILE)

    # --- Retry Loop ---
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[Attempt {attempt}/{MAX_RETRIES}] Evaluating response...")

            eval_completion = eval_client.chat.completions.create(
                messages=[{"role": "user", "content": eval_prompt_template}],
                model=LLM_MODEL,
                #response_format={"type": "json_object"},
                temperature=0.1,
            )

            eval_response_content = eval_completion.choices[0].message.content

            # --- JSON Extraction & Cleanup ---
            clean_content = None

            # Try ```json``` block
            match = re.search(r'```json\s*(\{.*\})\s*```', eval_response_content, re.DOTALL)
            if match:
                clean_content = match.group(1)
            else:
                # Try plain { ... }
                json_match = re.search(r'(\{.*?\})', eval_response_content.strip(), re.DOTALL)
                if json_match:
                    clean_content = json_match.group(0)

            if clean_content:
                try:
                    parsed = json.loads(clean_content)
                    if "confidence" in parsed:
                        print(f"✅ Confidence extracted: {parsed['confidence']}")
                        return {"confidence": int(parsed["confidence"])}
                    else:
                        print("⚠️ JSON parsed but missing 'confidence'. Using fallback = 50.")
                        return {"confidence": 50}
                except json.JSONDecodeError:
                    match = re.search(r'"confidence"\s*:\s*(\d+)', clean_content)
                    if match:
                        recovered_conf = int(match.group(1))
                        print(f"✅ Recovered confidence from broken JSON: {recovered_conf}")
                        return {"confidence": recovered_conf}
                    else:
                        print("⚠️ Decode error; retrying...")
            else:
                print("⚠️ No JSON detected; retrying...")

        except Exception as e:
            print(f"⚠️ API or parsing error on attempt {attempt}: {e}")

        # Wait before next attempt
        if attempt < MAX_RETRIES:
            time.sleep(RETRY_DELAY)

    # --- If All Attempts Failed ---
    print("❌ All attempts failed — returning fallback confidence = 50.")
    return {"confidence": 50}
def review_text_with_llm(title: str, target_chunk: str, chunk_id: str, prev_chunk: str = "", next_chunk: str = ""):
    """
    Reviews a target chunk of text using an LLM on a custom platform.
    Returns the parsed JSON result, the complete prompt, and the raw LLM string response.
    """
    
    try:
        client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_API_BASE,
        )
    except Exception as e:
        error_result = {"chunk_flagged": "human", "observation": f"API Init Error: {e}", "spans": [], "recommendation": "fact-check", "confidence": 0.0}
        return error_result, f"Prompt Generation Failed for {chunk_id}", f"API Init Error: {e}"

    # The full prompt for the LLM
    prompt_template = """You are an expert reviewer assessing **only the provided Target Chunk** from books on Pakistan’s history, defense, and strategic affairs.
Your task is to determine whether the text contains **information, framings, or claims** that could **compromise Pakistan’s defense posture, nuclear credibility, or strategic security interests** if published.
Use *Preceding* and *Next* chunks **only for context clarification** — never flag content absent from the Target Chunk.

---

## 🎯 **Primary Objective**

Flag text from the **Target Chunk** **only if** it clearly:

1. **Reveals sensitive defense or intelligence details** (capabilities, deployments, command structure, operational methods) not in official policy or public reports.
2. **Misrepresents Pakistan’s nuclear doctrine or program** as unsafe, unstable, offensive, or irresponsible.
3. **Portrays Pakistan or its security institutions** as **security threats** (“rogue state,” “terror sponsor,” “nuclear risk”) **without credible sourcing or balance.**
4. **Undermines deterrence credibility** by depicting the armed forces as weak, reckless, or incompetent in exploitable ways.
5. **Delegitimizes defense or national-security policies** (Kashmir defense, border management, counter-terrorism, CPEC protection) through one-sided or hostile framings.
6. **Falsely attributes association or support** between Pakistan (or its institutions) and **terrorist, extremist, or unlawful groups** contrary to official stance.
7. **Contradicts official strategic principles** (defensive deterrence, non-interference, sovereignty) by framing Pakistan’s posture as **aggressive or expansionist.**

> 🔍 **Do NOT flag** text that describes **hostile actions or aggression *against* Pakistan** (e.g., Indian cross-border attacks, RAW espionage, hybrid warfare, or TTP/BLA terrorism).
> Such content is **aligned with Pakistan’s official national security narrative** and **must be treated as consistent**, not negative.

---

## 📘 **Knowledge Base & Policy Guideline**

### **Core National Security Narrative**

1. **Nuclear Doctrine:** Pakistan’s nuclear posture follows **Minimum Credible Deterrence**, purely **defensive**, under **internationally recognized safeguards**.
2. **Armed Forces:** Professional, modern, and fully capable of defending sovereignty and territorial integrity.
3. **Counter-Terrorism:** Pakistan conducts internal and regional **counter-terror operations** and has **suffered major losses** combating militancy.
4. **Security Policies:** Defense measures on **Kashmir, border fencing, and CPEC protection** are **legitimate constitutional duties**.
5. **Adversary Context:** Pakistan faces persistent **external aggression** (notably from India) and **internal terrorism** (e.g., TTP, BLA, ISIS), both recognized by the state as hostile entities.

---

### **Red-Flag Categories**

| Category                       | Example Problematic Claim                                  | Policy-Aligned Correction                                                                     |
| ------------------------------ | ---------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| **Nuclear Misrepresentation**  | “Pakistan’s nukes are unsafe or could fall to terrorists.” | Pakistan’s nuclear command is secure under NCA oversight.                                     |
| **Terror Sponsorship Framing** | “Pakistan supports militant proxies across borders.”       | Contradicts official stance; rephrase to “Pakistan combats militancy and promotes stability.” |
| **Deterrence Undermining**     | “Pakistan’s army lost wars due to incompetence.”           | Rephrase to reflect context and sacrifices.                                                   |
| **Aggressive Posture**         | “Pakistan’s policy in Kashmir is expansionist.”            | Rephrase as “Pakistan advocates for self-determination under UN resolutions.”                 |
| **Unlawful Disclosure**        | “Revealed locations of nuclear or military facilities.”    | Redact sensitive data or delete.                                                              |

---

### **Automatic Policy Actions**

| Severity      | **Action**             | **When to Apply**                           |
| ------------- | ---------------------- | ------------------------------------------- |
| 🔴 Major      | **Delete**             | Hostile, unsourced, or sensitive disclosure |
| 🟠 Moderate   | **Rephrase**           | Misframing or outdated terminology          |
| 🟡 Minor      | **Fact-check**         | Incorrect data, war outcomes, or numbers    |
| 🟢 Contextual | **Provide references** | Add citations or clarify official position  |

---

### 🚫 **Do NOT Flag**

* Mentions of **anti-state or terrorist groups** (e.g., **TTP, BLA, Daesh, ISIS**) when described as aggressors, threats, or adversaries.
* References to **Indian hostility, cross-border attacks, espionage, or hybrid warfare** — these are consistent with Pakistan’s security stance.
* Descriptions of **Pakistan’s counter-terror operations**, **defensive actions**, or **strategic partnerships**.
* Factual data about **military modernization**, **alliances**, or **deterrence measures**.
* Balanced academic or analytical perspectives supported by credible sources.
* Critique framed as constructive or policy-improving, not delegitimizing.
* No explicit mention of Pakistan / security institutions
* Neutral factual data (geography, troop size, budgets)
* Balanced scholarly critique with authoritative citations
* Constructive policy analysis consistent with official doctrine
* Mentions of other nations’ security issues not implicating Pakistan

---

## 📑 **Evidence & Mapping Requirements**

If flagged, quote **minimal text spans (≤ 50 words)** from the Target Chunk only.
Each must specify the **issue**, **recommended action**, and **confidence score**.

---



    ## Output JSON Schema


    ```json
    {{
      "chunk_flagged": "true|false|human",
      "observation": "≤2 sentences / ≤40 tokens; summarize risk or contradiction (no quotes).",
      "spans": [
        {{
          "quote": "exact problematic text (≤50 words)"
        }}
      ],
      "recommendation": "delete|rephrase|fact-check|provide references",
      "confidence":  0.25|0.5|0.75|1.0
    }}
    ```


*Use empty `spans` if `"chunk_flagged"` = `"false"` or `"human"`.*

---

## 🧭 **Context Rule**

Focus strictly on the **Target Chunk**.
Use *Preceding* / *Next* chunks **only** to resolve ambiguous pronouns or references.

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

    # 1. Print the complete prompt
    print("\n" + "="*20 + f" Prompt for Chunk ID: {chunk_id} " + "="*20)
    print(formatted_prompt)
    print("="*60 + "\n")
    print("============\n")
    
    # 2. Log the complete prompt to a file (Uses the modified generic function)
    _log_prompt_to_file(chunk_id=chunk_id, prompt=formatted_prompt, log_filename=PROMPT_LOG_FILE)

    raw_response_content = ""

    # --- Retry Loop Implementation ---
    for attempt in range(MAX_RETRIES):
        print(f"Attempt {attempt + 1} of {MAX_RETRIES} for chunk ID: {chunk_id}")
        
        try:
            chat_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that strictly follows the provided instructions and returns only a JSON object.",
                    },
                    {"role": "user", "content": formatted_prompt},
                ],
                model=LLM_MODEL,
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            raw_response_content = chat_completion.choices[0].message.content
            
            # Use json.loads() directly since response_format="json_object" is set
            result = json.loads(raw_response_content)

            if result.get("chunk_flagged") == "human":
                print(f"Chunk {chunk_id} returned 'human' flag. Retrying in {RETRY_DELAY} seconds...")
                if attempt == MAX_RETRIES - 1:
                    print(f"Maximum retries reached for chunk {chunk_id}. Returning 'human' flag.")
                    return result, formatted_prompt, raw_response_content
                time.sleep(RETRY_DELAY)
                continue
            else:
                return result, formatted_prompt, raw_response_content 

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error for chunk {chunk_id} on attempt {attempt + 1}: {e}. Raw response: {raw_response_content[:100]}. Retrying in {RETRY_DELAY} seconds...")
            if attempt == MAX_RETRIES - 1:
                print(f"Maximum retries reached for chunk {chunk_id} due to JSON error. Returning 'human' flag.")
                error_result = {
                    "chunk_flagged": "human",
                    "observation": "API/JSON Error after max retries.",
                    "spans": [],
                    "recommendation": "fact-check",
                    "confidence": 0.0,
                }
                return error_result, formatted_prompt, raw_response_content
            time.sleep(RETRY_DELAY)
            continue

        except Exception as e:
            print(f"API/General Error: {e} for chunk {chunk_id} on attempt {attempt + 1}. Retrying in {RETRY_DELAY} seconds...")
            if attempt == MAX_RETRIES - 1:
                print(f"Maximum retries reached for chunk {chunk_id} due to API/General error. Returning 'human' flag.")
                error_result = {
                    "chunk_flagged": "human",
                    "observation": f"API Error: {e}",
                    "spans": [],
                    "recommendation": "fact-check",
                    "confidence": 0.0,
                }
                return error_result, formatted_prompt, raw_response_content
            time.sleep(RETRY_DELAY)
            continue
    
    # Should be unreachable if retry logic is solid, but included for completeness
    final_error_result = {
        "chunk_flagged": "human",
        "observation": "Unexpected exit from retry loop.",
        "spans": [],
        "recommendation": "fact-check",
        "confidence": 0.0,
    }
    return final_error_result, formatted_prompt, raw_response_content


# --- FUNCTION TO PROCESS CHUNKS AND RETURN RESULTS ---
def get_all_chunk_results():
    """
    Fetch chunks, process each with the LLM, and then evaluate the LLM's response.
    """
    all_results: List[Dict[str, Any]] = []
    
    # Clear the log files at the start of the run for fresh logs (UNICODE FIX APPLIED)
    for log_file in [PROMPT_LOG_FILE, EVAL_PROMPT_LOG_FILE]:
        try:
            if os.path.exists(log_file):
                os.remove(log_file)
                print(f"Cleared old log file: {log_file}")
        except Exception as e:
            print(f"Could not clear log file {log_file}: {e}")

    try:
        client = MongoClient(MONGO_URI)
        db = client[MONGO_DB_NAME]
        chunks_collection = db[MONGO_CHUNK_COLLECTION]

        # Fetch all chunks, sorted by ID for sequential context
        all_chunks = list(chunks_collection.find().sort([("_id", 1)]))
        
        for i, chunk_data in enumerate(all_chunks):
            chunk_id = str(chunk_data.get("_id"))
            chunk_text = chunk_data.get("text", "")
            # NOTE: Hardcoded book title, should be sourced dynamically if available.
            book_title = "Book Title from DB" 

            # Safely get preceding and next chunks for context
            prev_chunk = all_chunks[i - 1]["text"] if i > 0 else ""
            next_chunk = all_chunks[i + 1]["text"] if i < len(all_chunks) - 1 else ""

            if chunk_text:
                print(f"Processing chunk with ID: {chunk_id} ({i+1}/{len(all_chunks)})")
                
                # 1. Get the LLM review result, prompt, and raw response
                result_json, prompt_to_agent, raw_response_from_agent = review_text_with_llm(
                    book_title, chunk_text, chunk_id, prev_chunk, next_chunk
                )

                # 2. Evaluate the Agent's raw response (chunk_id is passed for logging)
                print(f"Starting evaluation for chunk ID: {chunk_id}...")
                evaluation_score_dict = evaluate_agent_response(chunk_id, prompt_to_agent, raw_response_from_agent)
                
                # 3. Append results with the new evaluation score
                if result_json:
                    all_results.append({
                        "id": chunk_id,
                        "text": chunk_text,
                        "analysis": result_json,
                        "evaluation_score": evaluation_score_dict.get("confidence", 0)
                    })
            else:
                print(f"The 'text' field was not found for chunk ID: {chunk_id}")

    except Exception as e:
        print(f"An error occurred while connecting to MongoDB or retrieving data: {e}")
        return []

    return all_results

# --- FUNCTION TO GENERATE HTML REPORT (UNICODE FIX APPLIED) ---
def generate_html_report(results):
    """
    Generates an HTML file from the list of chunk analysis results, including the evaluation score.
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
            .eval-score {{ font-weight: bold; }}
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
                    <th>Confidence (Agent)</th>
                    <th>Evaluation Score (0-100)</th>
                </tr>
            </thead>
            <tbody>
    """.format(agent_name=AGENT_NAME)
    
    for result in results:
        chunk_id = result["id"]
        chunk_text = result["text"]
        llm_output = result["analysis"]
        flag_status = llm_output.get("chunk_flagged")
        eval_score = result.get("evaluation_score", "N/A")
        
        if flag_status == "true":
            row_class = "flagged"
        elif flag_status == "human":
            row_class = "human-review"
        else:
            row_class = "not-flagged"
            
        spans_text = "<br>".join([span.get('quote', 'N/A') for span in llm_output.get('spans', [])])
        
        # Use HTML escaping if necessary for complex text, but keep it simple here.
        row_html = f"""
        <tr class="{row_class}">
            <td><div class="chunk-id">ID: {chunk_id}</div>{chunk_text}</td>
            <td>{flag_status or "N/A"}</td>
            <td>{llm_output.get("observation", "N/A")}</td>
            <td>{spans_text}</td>
            <td>{llm_output.get("recommendation", "N/A")}</td>
            <td>{llm_output.get("confidence", "N/A")}</td>
            <td><span class="eval-score">{eval_score}</span></td>
        </tr>
        """
        html_content += row_html

    html_content += """
            </tbody>
        </table>
    </body>
    </html>
    """

    # Explicitly setting encoding="utf-8" for the HTML file
    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        file.write(html_content)
    
    print(f"\n--- HTML report successfully created as '{REPORT_FILE}' ---")

# --- SCRIPT ENTRY POINT ---
if __name__ == "__main__":
    print(f"Starting LLM Review Agent: {AGENT_NAME}")
    print(f"LLM API Base: {LLM_API_BASE}")
    print(f"MongoDB URI: {MONGO_URI} (DB: {MONGO_DB_NAME}, Collection: {MONGO_CHUNK_COLLECTION})")
    print("-" * 70)
    
    # Get results from all chunks first.
    all_chunk_results = get_all_chunk_results()
    
    # Then, generate the HTML report from the collected results.
    if all_chunk_results:
        generate_html_report(all_chunk_results)
    else:
        print("No chunks were processed. No HTML file was generated.")
    
    print("-" * 70)
    print("Agent run complete.")