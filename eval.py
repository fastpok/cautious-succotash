from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT

from main import create_agent

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

load_dotenv()


# The test data consists of tuples where:
# - The first element is the test input query.
# - The second element is the reference output (expected result).
# - The third element is the reference SQL query. This parameter is not used in the test but is included for completeness.
test_data = [
    (
        "What is the average income of freelancers who accept cryptocurrency payments?",
        "The average income of freelancers who accept cryptocurrency payments is approximately $5,139.30.",
        "SELECT AVG(Earnings_USD) AS Average_Income FROM freelancer_earnings_bd WHERE Payment_Method = 'Crypto'",
    ),
    (
        "How many freelancers are from Asia?",
        "There are 281 freelancers from Asia.",
        "SELECT COUNT(*) AS NumberOfFreelancers FROM freelancer_earnings_bd WHERE Client_Region = 'Asia'",
    ),
    (
        "What is the highest hourly rate among freelancers in Europe?",
        "The highest hourly rate among freelancers in Europe is $98.83.",
        "SELECT MAX(Hourly_Rate) AS Highest_Hourly_Rate FROM freelancer_earnings_bd WHERE Client_Region LIKE '%Europe%'",
    ),
]

agent = create_agent("sqlite:///freelancer_earnings_bd.db")

correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    judge=llm,
)

passed_tests = 0

for i, (input, reference_output, reference_sql_query) in enumerate(test_data):
    output = agent.invoke({"input": input})["output"]
    eval_result = correctness_evaluator(
        inputs=input,
        outputs=output,
        reference_outputs=reference_output,
    )
    is_correct = eval_result["score"] == True
    passed_tests += is_correct
    print(f"Test case {i + 1}: {'PASSED' if is_correct else 'FAILED'}")

total_tests = len(test_data)
success_rate = (passed_tests / total_tests) * 100
print(f"Success rate: {success_rate:.2f}%")
