import re
import os

def extract_system_prompt_and_h2_headings(file_paths):
    """
    Extracts the initial instructional paragraph (the 'System Prompt') and 
    all H2 (##) Markdown-style headings from the main body of a list of files.

    Args:
        file_paths (list): A list of strings, where each string is the full
                           path to a file.
    """
    # REGEX for H2 headings: Matches lines starting with '##', followed by optional whitespace,
    # and strictly NOT followed by another '#'. This excludes H3 (###) and greater.
    h2_heading_pattern = r'^##[ \t\xa0]*(?!\#).*$'
    
    # Delimiter to find the end of the initial System Prompt block
    # It looks for a line containing only hyphens (---)
    delimiter_pattern = re.compile(r'^-+\s*$', re.MULTILINE)

    for file_path in file_paths:
        try:
            if not os.path.exists(file_path):
                print(f"\n--- ⚠️ File not found: {file_path} ---")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # --- 1. Extract System Prompt ---
            # Search for the first delimiter line (---)
            delimiter_match = delimiter_pattern.search(content)

            print(f"\n# --- Content from: {file_path} --- #")

            if delimiter_match:
                # The content before the delimiter is the System Prompt block
                system_prompt_text = content[:delimiter_match.start()].strip()
                # The content after the delimiter is the main instructional body
                main_content_body = content[delimiter_match.end():].strip()
                
                # Print the System Prompt
                print("## **System Prompt**")
                print(system_prompt_text)
                print("\n--- Extracted H2 Headings ---")
                
                # --- 2. Extract H2 Headings from Main Content ---
                # Find all H2 headings in the main instructional body
                h2_headings = re.findall(h2_heading_pattern, main_content_body, re.MULTILINE)
                
                if h2_headings:
                    for heading in h2_headings:
                        print(heading.strip())
                else:
                    print("No H2 (##) headings found in the main content body.")

            else:
                # Fallback if no delimiter is found
                print("Could not find the initial System Prompt block (content before the first '---').")
                
                # Still try to extract H2 headings from the whole file content
                print("\n--- Extracted H2 Headings (from full file) ---")
                h2_headings = re.findall(h2_heading_pattern, content, re.MULTILINE)
                if h2_headings:
                    for heading in h2_headings:
                        print(heading.strip())
                else:
                    print("No H2 (##) headings found.")


        except Exception as e:
            print(f"\n--- ❌ Error processing file {file_path}: {e} ---")

# --- Example Usage ---

# Define the list of file paths based on your uploaded files.
# NOTE: Replace these placeholder strings with the actual paths 
# where your files are saved on your system.
file_list = [
    "prompt1/Federal_Unity.txt",
    "prompt1/Instituitional_Integrity.txt",
    "prompt1/National_Security.txt",
    "prompt1/Rhetoric.txt",
    "prompt1/Historical.txt",
    "prompt1/Foreign_Policy.txt"
]

# Run the function
extract_system_prompt_and_h2_headings(file_list)