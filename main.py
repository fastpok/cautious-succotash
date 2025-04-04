import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

csv_filename = "freelancer_earnings_bd.csv"

system_message = "Answer in the same language in which the question is asked."


def load_csv_to_sqlite(csv_path):
    base_filename = os.path.splitext(csv_path)[0]
    sql_uri = "sqlite:///" + base_filename + ".db"
    engine = create_engine(sql_uri)
    df = pd.read_csv(csv_path)
    df.to_sql(name=base_filename, con=engine, if_exists="replace")
    return sql_uri


def create_agent(sql_uri):
    db = SQLDatabase.from_uri(sql_uri)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent = create_sql_agent(
        llm=llm,
        agent_type="tool-calling",
        toolkit=toolkit,
        verbose=False,
    )
    return agent


def run_agent(agent):
    print("Type 'exit' to quit or press Ctrl+C to exit")
    try:
        while True:
            prompt = input("Enter a prompt: ")
            if prompt.lower() == "exit":
                print("Exiting...")
                break
            else:
                result = agent.invoke(
                    [
                        SystemMessage(content=system_message),
                        HumanMessage(content=prompt),
                    ]
                )
                print(result["output"])
    except KeyboardInterrupt:
        print("\nExiting...")


def main():
    load_dotenv()
    sql_uri = load_csv_to_sqlite(csv_filename)
    agent = create_agent(sql_uri)
    run_agent(agent)


if __name__ == "__main__":
    main()
