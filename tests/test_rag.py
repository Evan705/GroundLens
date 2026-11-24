
from rag import retrieve_context
from rag import generate_answer
from deepeval.test_case import LLMTestCase
from deepeval import assert_test

# Verify that the correct annual leave information is retrieved.
def test_annual_leave_retrieval():
    question = "How many annual leave days are available?"
    context = retrieve_context(question)
    assert "20 days" in context


# Verify that the correct sick leave information is retrieved.
def test_sick_leave_retrieval():
    question = "How many sick leaves are available?"
    context = retrieve_context(question)
    assert "10 days" in context


# Verify that the correct parental leave information is retrieved.
def test_parental_leave():
    question = "How many parental leaves are available?"
    context = retrieve_context(question)
    assert "12 weeks" in context


# Verify that the correct work-from-home allowance is retrieved.
def test_wfh_retrieval():
    question = "How many work from home days are available?"
    context = retrieve_context(question)
    assert "3 days per week" in context


# Verify that the travel policy includes the manager approval requirement.
def test_travel_policy_retrieval():
    question = "Give me Travel policy details"
    context = retrieve_context(question)
    assert "manager approval" in context


# Verify that unrelated questions return the expected fallback message.
def test_unrelated_question():
    question = "How many bereavement leaves are available?"
    context = retrieve_context(question)
    assert "No relevant information found in the document." in context


# Verify that alternative wording such as "vacation" retrieves annual leave.
def test_vacation_leave_retrieval():
    context = retrieve_context("How many vacation days can I take?")
    assert "20 days" in context


# Verify that "medical leave" retrieves the sick leave information.
def test_medical_leave_retrieval():
    context = retrieve_context("What is my medical leave entitlement?")
    assert "10 days" in context


# Verify that "remote work" retrieves the work-from-home policy.
def test_remote_work_retrieval():
    context = retrieve_context("Can I work remotely?")
    assert "3 days per week" in context

    """Check whether expected text exists in retrieved context."""

    if isinstance(context, list):
        context = " ".join(context)

    assert expected_text.lower() in context.lower()
    context = retrieve_context("Do I need permission for a business trip?")
    assert "manager approval" in context

    question = "What are the annual and sick leave entitlements?"

    expected_answer = """
    Employees are entitled to 20 days of annual leave
    and 10 days of sick leave per year.
    """

    context = retrieve_context(question)

    if isinstance(context, (list, tuple)):
        retrieval_context = [str(item) for item in context]
    else:
        retrieval_context = [str(context)]

    answer = generate_answer(question, context)

    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected_answer,
        retrieval_context=retrieval_context
    )

    assert_test(
        test_case
    )