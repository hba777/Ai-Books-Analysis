import os
import subprocess
import json
import re
from groq import Groq
from dotenv import load_dotenv
from pymongo import MongoClient
import sys

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("❌ Error: GROQ_API_KEY not found in environment variables.")
    print("Please create a .env file with the line: GROQ_API_KEY='your_api_key'")
    sys.exit(1)

def get_response_from_groq(prompt):
    """
    Sends a prompt to the Groq API and returns the response.
    """
    client = Groq(api_key=GROQ_API_KEY)
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"❌ An error occurred during the Groq API call: {e}"

def store_response_in_db(agent_name, response_data):
    """
    Stores the Groq response in MongoDB as JSON.
    """
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client["groq_responses_db"]
        collection = db["responses1"]

        document_to_store = {
            "agent_name": agent_name,
            "chunk_flagged": response_data.get("chunk_flagged"),
            "observation": response_data.get("observation"),
            "recommendation": response_data.get("recommendation"),
            "spans": response_data.get("spans", [])
        }

        insert_result = collection.insert_one(document_to_store)
        print(f"✅ Successfully stored response in MongoDB with ID: {insert_result.inserted_id}")
        return True
    except Exception as e:
        print(f"❌ Error storing data in MongoDB: {e}")
        return False

def extract_json_from_text(text):
    """
    Attempts to extract JSON from Groq response text.
    """
    try:
        json_start = text.find('{')
        json_end = text.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_string = text[json_start:json_end]
            return json.loads(json_string)
    except json.JSONDecodeError:
        pass
    return {
        "chunk_flagged": None,
        "observation": None,
        "recommendation": None,
        "spans": []
    }

def split_chunk_into_lines(text):
    """
    Splits a long chunk into multiple logical lines for better Groq parsing.
    Splits first on periods, then on commas if the line is too long.
    """
    sentences = re.split(r'(?<=[.]) +', text)
    lines = []
    for s in sentences:
        if len(s) > 120:  # threshold for long sentences
            lines.extend([x.strip() for x in s.split(",") if x.strip()])
        else:
            lines.append(s.strip())
    return "\n".join(lines)

def format_long_text_as_target_chunk(long_text):
    """
    Converts any long multi-paragraph text into a single-line target_chunk
    with references preserved.
    """
    # Remove extra line breaks and whitespace
    text = re.sub(r'\s+', ' ', long_text).strip()
    # Ensure references like 190, 191 are separated by semicolons
    text = re.sub(r'(\d{3} )', r'; \1', text)
    return text

if __name__ == "__main__":
    agent_name = "Institutional Integrity"
    book_title = "The History of Pakistan"

    # Example raw long text (replace with your own)
    long_text = """70 Integrity Assessment Special Inspector General for Afghanistan Reconstruction (SIGAR), which oversaw the reconstruction in the country, was established in 2008 by the US government.167 According to the SIGAR quarterly report of 2019, the strength management and payroll system followed by the MoD in Afghanistan provided opportunities for corruption and mall practices. There have been ghost soldiers, payrolls, and payments for many years. SIGAR investigators unearthed that government officials, both at central and provincial police and military headquarters throughout the country, made fake payroll records and were receiving payments against non- existent employees. However, MoD and MoI rejected the claims.168 Another alarming phenomenon during the initial days of the Karzai tenure was the theft and selling of arms and ammunition by the police and military soldiers.169 At times these were found in the hands of the Taliban and other outlaws or sold in open markets. Violations by ANA Human rights violations by the state organs, especially the police, were a common phenomenon, and Afghans are otherwise famous for brutalities against opponents and enemies. During wars, this attitude gets multiplied as the law enforcement agencies in the garb of law enforcers commit atrocities and excesses. Only in 2019, according to UNAMA, over one thousand civilian casualties occurred, mainly by the Afghan military and the air force, with around 400 deaths and 700 injuries.170 Air bombings resulted in deaths to the civil populace, which repeatedly happened; one of the most criticized was in Imam Sahib, Kunduz, in northern Afghanistan, where 11 civilians were killed. ANA’s long-range artillery bombing and mortar shelling also resulted in civilian casualties. Forced disappearances, deaths during detention, arbitrary detention, and torture by state agencies was other serious concern and violation that attracted criticism and lowered the government's popularity. As per the UNAMA report171 in two years (2017-2018), dozens of ANA detainees were interviewed who accepted excesses, and 36 percent reported torture, ill-treatment, sexual harassment of young boys, and severe beating during detention. 167 “Special Inspector General for Afghanistan Reconstruction (SIGAR).” 2021. About Sigar. Accessed August 19. https://www.sigar.mil/about/. 168 SIGAR Quarterly Report, “Special Inspector General for SIGAR Afghanistan Reconstruction OCT 30 2019.” sigar.mil/pdf/quarterly reports, 2020. https://www.sigar.mil/pdf/quarterlyreports/2019-10-30qr.pdf. p-83 169 Ibid 170 European Asylum Support Office.” Afghanistan State Structures and Security Forces: Country of Origin Information Report”. LU: Publications Office, 2020. https://data.europa.eu/doi/10.2847/115002. 171 United Nations Assistance Mission in Afghanistan, Rep. “Treatment of Conflict-Related Detainees in Afghanistan: Preventing Torture and Ill-Treatment under the Anti-Torture Law.” https://unama.unmissions.org, 2020. p 12-21"""
    # Convert raw text into single-line target_chunk
    target_chunk = format_long_text_as_target_chunk(long_text)

    # Split long single-line chunk into multiple lines for better Groq parsing
    formatted_chunk = split_chunk_into_lines(target_chunk)

    # Run the prompt generation script
    command = ["python", "generate_prompt.py", agent_name, book_title, formatted_chunk]

    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True, encoding='utf-8')
        prompt = result.stdout.strip()

        if "❌" in prompt:
            print(prompt)
        else:
            print("Generated Prompt:\n" + prompt)
            groq_response_text = get_response_from_groq(prompt)
            print("Raw Groq Response:\n" + groq_response_text)

            groq_response_json = extract_json_from_text(groq_response_text)
            store_response_in_db(agent_name, groq_response_json)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error running the prompt generation script: {e.stderr}")
    except FileNotFoundError:
        print("❌ Error: The 'generate_prompt.py' file was not found.")
        print("Please make sure both files are in the same directory.")
