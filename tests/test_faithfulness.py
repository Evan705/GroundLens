
import pytest
from rag import retrieve_context, generate_answer
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRelevancyMetric,HallucinationMetric
from deepeval.models import OllamaModel


# Configure Ollama as the local LLM evaluator for DeepEval.
evaluator = OllamaModel(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
    temperature=0
)


# Configure the faithfulness metric to check whether
# generated answers are supported by the retrieved context.
faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model=evaluator
)


# Configure the answer relevancy metric to check whether
# generated answers are relevant to the user's question.
relevancy_metric = AnswerRelevancyMetric(
    threshold=0.7,
    model=evaluator
)

contextual_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    model = evaluator,
    include_reason=True
)

hallucination_metric = HallucinationMetric(
    threshold=0.5,
    model=evaluator,
    include_reason=True
)


# Parameterize the test to execute the same evaluation
# for multiple HR policy questions.
@pytest.mark.parametrize("question", [
    "How many annual leave days are available?",
    "How many sick leave days are available?",
    "How many weeks of parental leave are available?",
    "How many days can employees work from home?",
    "What approval is required before booking business travel?"
])
def test_leave_faithfulness(question):

    # Retrieve relevant information from the HR policy document.
    context = retrieve_context(question)

    # Generate an answer using the retrieved context and local LLM.
    answer = generate_answer(question, context)

    # Create a DeepEval test case containing the question,
    # generated answer, and retrieved context.
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        retrieval_context=[str(context)],
        context=[str(context)]
    )

    # Evaluate the generated answer using both configured metrics.
    assert_test(test_case, [faithfulness, relevancy_metric, contextual_relevancy,hallucination_metric])

    # Print details to help inspect the evaluation inputs and output.
    print("QUESTION:", question)
    print("RETRIEVED CONTEXT:", context)
    print("GENERATED ANSWER:", answer)
