# text_classifier.py
from transformers import pipeline

def initialize_classifier():
    """
    Initializes and returns the zero-shot classification pipeline.
    This function helps avoid reloading the model multiple times if
    you're classifying many texts.
    """
    print("Loading zero-shot classifier (facebook/bart-large-mnli)... This may take a moment.")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print("Classifier loaded successfully!")
    return classifier

# Initialize the classifier once when the module is imported
classifier = initialize_classifier()

# Define your labels
labels = [
    "historical or military event",
    "name of army unit or regiment",
    "military operation",
    "specific date, timelines or year",
    "military rank or officer name",
    "martyrdom or sacrifice story",
    "weapon, vehicle, or equipment mention",
    "award, medal, or decoration",
    "military training or exercise",
    "operational statistics or mission facts",
    "bilateral realtions",
    "institutions",
    "general or unrelated text"
]

# Hypothesis template
template = "This text is about {}."

def classify_text(text: str) -> dict:
    """
    Classifies a given text chunk into one of the predefined military-related categories
    using a zero-shot transformer model. It always returns the top predicted label
    and all confidence scores.

    Args:
        text (str): The text chunk to be classified.

    Returns:
        dict: A dictionary containing the original text, predicted label,
              its confidence score, and all other labels' confidence scores.
    """
    if not text:
        return {
            "text": text,
            "predicted_label": "Empty input text",
            "confidence": 0.0,
            "all_scores": {}
        }

    # Perform the zero-shot classification
    result = classifier(text, candidate_labels=labels, hypothesis_template=template, multi_label=False)

    # The result contains sorted labels and scores, with the highest confidence first
    top_label = result['labels'][0]
    top_score = result['scores'][0]

    return {
        "text": text,
        "predicted_label": top_label,
        "confidence": round(top_score, 3),
        # Round all scores for cleaner output
        "all_scores": {label: round(score, 3) for label, score in zip(result['labels'], result['scores'])}
    }

# This __main__ block is for testing the classifier independently
if __name__ == "__main__":
    print("\n--- Zero-Shot Classification Examples ---")

    test_chunks = [
        "Who are u?",
        "The Battle of Gettysburg was a turning point in the Civil War.",
        "On December 7, 1941, Pearl Harbor was attacked, a day that will live in infamy.",
        "General Patton led the Third Army with decisive action.",
        "The 101st Airborne Division is known as the 'Screaming Eagles' for its distinctive patch.",
        "Operation Desert Storm began in January 1991, aiming to liberate Kuwait.",
        "He was posthumously awarded the Medal of Honor for his extraordinary bravery.",
        "The M1 Abrams tank is a powerful main battle tank, a cornerstone of armored warfare.",
        "Private Miller made the ultimate sacrifice for his country during the intense firefight.",
        "The soldiers underwent rigorous basic training, preparing for deployment.",
        "The Pentagon confirmed the troop deployment, citing security concerns.",
        "The sky is blue and the grass is green.",
        "World War II ended in 1945, reshaping global politics.",
        "The 82nd Airborne conducted a large-scale parachute jump exercise last week.",
        "Lieutenant Colonel Smith commanded the battalion with distinction.",
        "The Tet Offensive was a major campaign during the Vietnam War.",
        "The Victoria Cross is the highest military decoration awarded for valor in the British armed forces.",
        "They used an AR-15 rifle, a common semi-automatic firearm.",
        "His sacrifice on the battlefield will never be forgotten by his comrades.",
        "The information has been verified by multiple official sources and is considered factual.",
        "I went to the grocery store today and bought some milk.",
        "The D-Day landings occurred on June 6, 1944, marking the beginning of the end for Nazi Germany.",
        "The USS Arizona was sunk during the attack on Pearl Harbor.",
        "Sergeant Johnson was promoted for his exemplary leadership skills.",
        "The Marine Corps is a branch of the US Armed Forces, renowned for its amphibious capabilities.",
        "Exercise Red Flag tests combat readiness in a realistic air combat environment.",
        "He received a Purple Heart after being wounded in action.",
        "The Battle of Antietam was fought in 1862, known for being the bloodiest single-day battle in American history.",
        "The 75th Ranger Regiment is an elite special operations force of the United States Army.",
        "Operation Enduring Freedom, launched in response to 9/11, lasted for many years.",
        "The brave soldiers defended their position to the last man, showing incredible resolve.",
        "The data confirms the enemy's movements, providing crucial intelligence.",
        "The new fighter jet has advanced stealth capabilities and superior maneuverability.",
        "The year is 2024, and technology continues to advance rapidly.",
        "The drill involved urban warfare simulations, preparing troops for diverse environments.",
        "Captain Jones led the charge, inspiring his troops to push forward.",
        "The Nobel Prize is a prestigious international award given in several categories.",
        "She enjoyed reading a fictional book about ancient civilizations.",
        "The Normandy landings were a pivotal moment in World War II, opening a new front.",
        "The 1st Cavalry Division is known for its mobility and combines air and ground elements.",
        "Operation Iraqi Freedom was a large-scale military intervention that began in 2003.",
        "His memory lives on through his heroic actions and the lives he saved.",
        "The information presented is factually accurate and has been vetted by experts.",
        "They deployed armored personnel carriers to transport troops safely through hostile territory.",
        "The event took place on 2023-01-15, a clear winter day.",
        "The platoon practiced close-quarters combat, honing their skills in confined spaces.",
        "General Eisenhower commanded the Allied forces during World War II, a testament to his leadership.",
        "He earned the Bronze Star for his meritorious service in a combat zone.",
        "I like pizza with extra cheese and pepperoni."
    ]

    for i, chunk in enumerate(test_chunks):
        classification_result = classify_text(chunk)
        print(f"\n--- Test Case {i+1} ---")
        print(f"Input Text: \"{classification_result['text']}\"")
        print(f"Predicted Label (Highest Confidence): \"{classification_result['predicted_label']}\"")
        print(f"Confidence for Predicted Label: {classification_result['confidence']}")
        print(f"All Label Scores: {classification_result['all_scores']}")
        print("-" * 30)

    print("\n--- Interactive Classification ---")
    while True:
        user_input = input("Enter text to classify (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        result = classify_text(user_input)
        print(f"Input: \"{result['text']}\"")
        print(f"Predicted Label (Highest Confidence): \"{result['predicted_label']}\"")
        print(f"Confidence for Predicted Label: {result['confidence']}")
        print(f"All Label Scores: {result['all_scores']}")
        print("-" * 30)