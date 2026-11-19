
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
    """Retrieve relevant HR policy information."""

    # Load the document and split it into individual policy sections.
    document = load_document()
    sections = document.split("\n\n")

    # Map different question keywords to their corresponding policy sections.
    # This uses keyword-based retrieval, not semantic or vector search.
    keywords = {
        "annual": "Annual Leave",
        "vacation": "Annual Leave",
        "paid leave": "Annual Leave",

        "sick": "Sick Leave",
        "medical leave": "Sick Leave",

        "parental": "Parental Leave",
        "maternal": "Parental Leave",
        "paternity": "Personal Leave",

        "work from home": "Work From Home",
        "remote": "Work From Home",
        "wfh": "Work From Home",

        "travel": "Travel Policy",
        "business trip": "Travel Policy"
    }

    # Normalize the question to make keyword matching case-insensitive.
    question = question.lower().strip()

    # Check whether any keyword matches the question.
    for keyword, section_name in keywords.items():
        if keyword in question:

            # Find and return the matching policy section.
            for section in sections:
                if section.lower().startswith(section_name.lower()):
                    return section

    # Return a fallback message when no matching section is found.
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
