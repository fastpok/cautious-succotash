import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.agent_toolkits import create_sql_agent

csv_filename = "freelancer_earnings_bd.csv"

base_filename = os.path.splitext(csv_filename)[0]
sql_uri = "sqlite:///" + base_filename + ".db"


def load_csv_to_sqlite(csv_path):
    engine = create_engine(sql_uri)
    df = pd.read_csv(csv_path)
    df.to_sql(name=base_filename, con=engine, if_exists="replace")


def create_agent():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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
    print("Type 'exit' to quit")
    while True:
        prompt = input("Enter a prompt: ")
        if prompt.lower() == "exit":
            print("Exiting...")
            break
        else:
            result = agent.invoke({"input", prompt})
            print(result["output"])


def main():
    load_dotenv()
    load_csv_to_sqlite(csv_filename)
    agent = create_agent()
    run_agent(agent)


if __name__ == "__main__":
    main()
