
import ollama
from pathlib import Path

# Define the location of the HR policy document.
# Path(__file__).parent refers to the directory containing this Python file.
POLICY_FILE = Path(__file__).parent / "data" / "hr_policy.txt"


# Load the HR policy document from the data folder.
def load_document():
    """Load HR policy document."""
    return POLICY_FILE.read_text(encoding="utf-8")


# Retrieve the relevant policy section based on keywords in the question.

def retrieve_context(question):
    """Retrieve all relevant HR policy sections."""

    document = load_document()
    sections = document.split("\n\n")

    keywords = {
        "annual": "Annual Leave",
        "vacation": "Annual Leave",
        "paid leave": "Annual Leave",

        "sick": "Sick Leave",
        "medical leave": "Sick Leave",

        "parental": "Parental Leave",
        "maternal": "Parental Leave",
        "paternity": "Parental Leave",

        "work from home": "Work From Home",
        "remote": "Work From Home",
        "wfh": "Work From Home",

        "travel": "Travel Policy",
        "business trip": "Travel Policy"
    }

    question = question.lower().strip()

    retrieved_sections = []

    # Identify all matching policy sections.
    for keyword, section_name in keywords.items():

        if keyword in question:

            for section in sections:

                if section.lower().startswith(section_name.lower()):

                    if section not in retrieved_sections:
                        retrieved_sections.append(section)

    # Return all matching sections.
    if retrieved_sections:
        return retrieved_sections

    return "No relevant information found in the document."

# Generate an answer using the retrieved policy context and a local LLM.
def generate_answer(question, context):
    """Generate an answer using HR policy context."""

    # Construct a prompt that restricts the LLM to the provided context.
    prompt = f"""
    You are an HR Policy Assistant.

    Answer the question in one complete sentence
    using only the information in the context.

    If the answer is not available in the context, say:
    Information not available in the policy.

    Context: {context}
    Question: {question}
    Answer: """

    # Send the prompt to the locally running Ollama model.
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.message.content.strip()
