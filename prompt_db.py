import os
import re

FOLDER = "prompt1"

EXCLUDE_KEYWORDS_2HASH = [
    "evidence",
    "output format",
    "context rule",
    "inputs"
]

EXCLUDE_KEYWORDS_3HASH = [
    "do not flag",
    "book title",
    "preceding chunk",
    "target chunk",
    "next chunk"
]

def normalize(title):
    """
    Normalize heading:
    - lowercase
    - remove markdown bold/italic (** or __)
    - remove backslashes
    - remove colons and other punctuation
    - remove literal '\n', '\r'
    - replace underscores with space
    - collapse multiple spaces
    """
    # Remove markdown bold/italic
    title = re.sub(r'(\*|_){1,3}', '', title)
    # Remove backslash
    title = title.replace('\\', '')
    # Underscore → space
    title = title.replace('_', ' ',)
    # Remove colon and potential trailing newline found in your printout
    title = title.replace(':', '').replace('\\n', '')
    # Remove literal newlines
    title = title.replace('\n', '').replace('\r', '')
    # Remove other punctuation
    title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
    # Collapse spaces
    title = re.sub(r'\s+', ' ', title)
    
    return title.lower().strip()

def should_exclude(title, level):
    norm = normalize(title)
    if level == 2:
        return any(kw in norm for kw in EXCLUDE_KEYWORDS_2HASH)
    elif level == 3:
        return any(kw in norm for kw in EXCLUDE_KEYWORDS_3HASH)
    return False

def parse_prompt_file(text):
    """
    Splits the text into System Prompt (initial block) and content sections.
    """
    # Find the start of the first heading (any level: #, ##, ###)
    first_heading_match = re.search(r'^(#+)\s*(.+)$', text, re.MULTILINE)
    
    system_prompt = ""
    sections = []
    
    if first_heading_match:
        # 1. Extract System Prompt
        # Text from start up to the start of the first heading
        system_prompt = text[:first_heading_match.start()].strip()
        
        # Now process the rest of the text for sections
        content_text = text[first_heading_match.start():]
        
        # Find all ## and ### headings in the content_text
        pattern = re.compile(r'^(#{2,3})\s*(.+)$', re.MULTILINE)
        matches = list(pattern.finditer(content_text))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            # Get the raw title text
            title = match.group(2).strip()

            if should_exclude(title, level):
                continue

            start = match.end()
            # Find the start of the next heading or the end of the text
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content_text)
            content = content_text[start:end].strip()
            sections.append((level, title, content))

    else:
        # If no headings are found, the whole file is the system prompt
        system_prompt = text.strip()

    return system_prompt, sections

def read_all_files(folder):
    results = {}
    if not os.path.exists(folder):
        print(f"❌ Error: Folder '{folder}' not found. Please create it and add .txt files.")
        return results

    for file_name in os.listdir(folder):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder, file_name)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                # Use the new function to get system prompt and sections
                system_prompt, sections = parse_prompt_file(text)
                
                # Store the results
                results[file_name] = {
                    "system_prompt": system_prompt,
                    "sections": sections
                }
                print(f"✅ Processed: {file_name}")
            except Exception as e:
                print(f"❌ Failed to process {file_name}: {e}")

    return results

if __name__ == "__main__":
    all_sections_data = read_all_files(FOLDER)

    # Print summary
    for file, data in all_sections_data.items():
        print(f"\n📘 File: {file}")
        
        # Print System Prompt as requested
        print("  SYSTEM PROMPT:")
        print(f"  ---{'-' * len(data['system_prompt'].splitlines()[0])}---")
        # Print the system prompt, indented
        for line in data['system_prompt'].splitlines():
             print(f"  | {line}")
        print(f"  ---{'-' * len(data['system_prompt'].splitlines()[0])}---")

        
        print("\n  **Extracted Content Sections (Non-Excluded):**")
        sections = data['sections']
        if not sections:
            print("  (No non-excluded sections found)")
        for level, title, _ in sections:
            print(f"  {'#' * level} {title}")