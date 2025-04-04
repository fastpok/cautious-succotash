from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from openevals import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_HELPFULNESS_PROMPT

from main import create_agent, system_message

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

load_dotenv()

# The test data consists of tuples where:
# - The first element is the test input query.
# - The second element is the reference output (expected result).
# - The third element is the reference SQL query. This parameter is not used in the test but is included for completeness.
test_data = [
    (
        """Насколько выше доход у фрилансеров, принимающих оплату в криптовалюте, по сравнению с другими способами оплаты?""",
        """
            Средний доход фрилансеров, принимающих оплату в криптовалюте, составляет примерно 5139.30 долларов США. Для сравнения, средний доход фрилансеров, использующих другие методы оплаты, составляет:

            - Банковский перевод: 5019.96 долларов США
            - PayPal: 4976.69 долларов США
            - Мобильный банкинг: 4923.65 долларов США

            Таким образом, фрилансеры, принимающие оплату в криптовалюте, имеют наивысший средний доход по сравнению с другими способами оплаты.
        """,
        """
            SELECT 
                Payment_Method, 
                AVG(Earnings_USD) AS Average_Earnings 
            FROM 
                freelancer_earnings_bd 
            GROUP BY 
                Payment_Method 
            ORDER BY 
                Average_Earnings DESC 
            LIMIT 10;
        """,
    ),
    (
        """Как распределяется доход фрилансеров в зависимости от региона проживания?""",
        """
            Доход фрилансеров в зависимости от региона проживания распределяется следующим образом (средний доход в USD):

            1. Канада: 5350.13
            2. Азия: 5172.28
            3. Великобритания: 5047.09
            4. Австралия: 4966.10
            5. Европа: 4890.53
            6. США: 4872.95
            7. Ближний Восток: 4870.82

            Эти данные показывают, что фрилансеры из Канады имеют наивысший средний доход.
        """,
        """SELECT Client_Region, AVG(Earnings_USD) as Average_Earnings FROM freelancer_earnings_bd GROUP BY Client_Region ORDER BY Average_Earnings DESC LIMIT 10;""",
    ),
    (
        """Какой процент фрилансеров, считающих себя экспертами, выполнил менее 100 проектов?""",
        """Процент фрилансеров, считающих себя экспертами и выполнивших менее 100 проектов, составляет примерно 33.85%.""",
        """
            SELECT 
                (SELECT COUNT(*) FROM freelancer_earnings_bd WHERE Experience_Level = 'Expert' AND Job_Completed < 100) * 100.0 / 
                (SELECT COUNT(*) FROM freelancer_earnings_bd WHERE Experience_Level = 'Expert') AS percentage
            ;
        """,
    ),
    (
        """Сколько суммарно заработали 10 самых успешных фрилансеров?""",
        """Суммарно 10 самых успешных фрилансеров заработали 9784255 долларов.""",
        """SELECT sum(Earnings_USD) FROM freelancer_earnings_bd ORDER BY Earnings_USD DESC LIMIT 10;""",
    ),
    (
        """В какой категории работает больше всего фрилансеров?""",
        """Больше всего фрилансеров работает в категории Graphic Design.""",
        """SELECT Job_Category, COUNT(*) AS freelancer_count FROM freelancer_earnings_bd GROUP BY Job_Category ORDER BY freelancer_count DESC LIMIT 1;""",
    ),
]

agent = create_agent("sqlite:///freelancer_earnings_bd.db")

# Evaluates: Final output vs. input + reference answer
# Goal: Measure "how similar/correct is the generated answer relative to a ground-truth answer"
# Requires reference: Yes
correctness_evaluator = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    feedback_key="correctness",
    judge=llm,
)

# Evaluates: Final output vs. input
# Goal: Measure "how well does the generated response address the initial user input"
# Requires reference: No, because it will compare the answer to the input question
helpfulness_evaluator = create_llm_as_judge(
    prompt=RAG_HELPFULNESS_PROMPT,
    feedback_key="helpfulness",
    judge=llm,
)

total_tests = len(test_data)
eval_results = {}

for i, (input, reference_output, reference_sql_query) in enumerate(test_data):
    output = agent.invoke(
        [
            SystemMessage(content=system_message),
            HumanMessage(content=input),
        ]
    )["output"]
    correctness_eval_result = correctness_evaluator(
        inputs=input,
        outputs=output,
        reference_outputs=reference_output,
    )
    helpfulness_eval_result = helpfulness_evaluator(
        inputs=input,
        outputs=output,
    )
    eval_results[i] = {
        "correctness": correctness_eval_result["score"],
        "helpfulness": helpfulness_eval_result["score"],
    }
    print(f"Test case {i + 1} of {total_tests}: {eval_results[i]}")

# Calculate the average scores for each evaluator
avg_scores = {
    key: sum(result[key] for result in eval_results.values()) / total_tests
    for key in ["correctness", "helpfulness"]
}

print(f"Average Scores: {avg_scores}")
