import argparse
import glob
import json
import os
import re
import sqlite3
import time
from typing import List, Optional

import openai
from openai.error import Timeout as OpenAITimeout
import pandas as pd
from requests.exceptions import Timeout as RequestsTimeout
import sqlglot
from sqlglot import expressions, Expression
import tiktoken

# Set your OpenAI API key here
openai.api_key = 'your-api-key-here'

parser = argparse.ArgumentParser(description="Set command-line arguments.")
parser.add_argument(
    "--set", type=str, default="train", help="Set value for SET variable."
)
parser.add_argument(
    "--json", type=str, default="train.json", help="Set value for JSON_FILE variable."
)
parser.add_argument(
    "--db", type=str, default="train_databases", help="Set value for DB_DIR variable."
)
parser.add_argument(
    "--dryrun", type=bool, default=False, help="Set value for DRYRUN variable."
)
args = parser.parse_args()

SET = args.set
JSON_FILE = os.path.join(SET, args.json)
DB_DIR = os.path.join(SET, args.db)

if not os.path.exists(DB_DIR):
    raise ValueError(f"DB_DIR {DB_DIR} does not exist.")
if not os.path.exists(JSON_FILE):
    raise ValueError(f"JSON_FILE {JSON_FILE} does not exist.")

dryrun = args.dryrun

DEFAULT_MODEL = "gpt-3.5-turbo"  # Fallback model if the custom model is not accessible
CUSTOM_MODEL = "ft:gpt-3.5-turbo-0613:mercator:dev:8GFLwXLG"  # Replace with your custom model

encoding = tiktoken.get_encoding("cl100k_base")

def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def get_create_table_and_data(db_path: str, num_rows: int = 5) -> List[str]:
    MAX_TOKENS = 2550  # The limit required so OpenAI doesn't complain after we reformat
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    while num_rows >= 0:
        # Query the sqlite_master table to get the CREATE TABLE statements
        cursor.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        tables = cursor.fetchall()

        output_statements = []

        for table_name, create_statement in tables:
            # "INTEGER" -> "INT"
            create_statement = create_statement.replace("INTEGER", "INT")

            # remove comments
            create_statement = re.sub(
                r"--.*$", "", create_statement, flags=re.MULTILINE
            )
            create_statement = "\n".join(
                [line for line in create_statement.split("\n") if line.strip()]
            )

            # Condense whitespace
            create_statement = " ".join(create_statement.split())

            # First, add the create statement
            output_statements.append(create_statement + ";")

            # Fetch sample data
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT ?", (num_rows,))
            sample_rows = cursor.fetchall()

            # For each row, create an INSERT INTO statement
            for row in sample_rows:
                formatted_values = []
                for idx, value in enumerate(row):
                    if isinstance(value, str):
                        formatted_value = value.replace("\n", " ")
                        formatted_value = formatted_value.replace("'", '"')
                        formatted_value = formatted_value[:100]
                        formatted_values.append(f"'{formatted_value}'")
                    elif value is None:
                        formatted_values.append("NULL")
                    else:
                        formatted_values.append(str(value))
                values_str = ",".join(formatted_values)

                # Check if table_name contains a space or dash and wrap it in double quotes if it does
                if " " in table_name or "-" in table_name:
                    formatted_table_name = f'"{table_name}"'
                else:
                    formatted_table_name = table_name

                insert_statement = (
                    f"INSERT INTO {formatted_table_name} VALUES ({values_str});"
                )
                output_statements.append(insert_statement)

        msgs = [{"role": "user", "content": "\n".join(output_statements)}]
        token_count = num_tokens_from_messages(msgs)

        if token_count < MAX_TOKENS:
            cursor.close()
            conn.close()
            return output_statements
        elif num_rows > 0:
            num_rows -= 1
            continue
        else:
            final_statements = []
            for statement in output_statements:
                final_statements.append(statement)
                msgs = [{"role": "user", "content": "\n".join(final_statements)}]
                token_count = num_tokens_from_messages(msgs)

                if token_count > MAX_TOKENS:
                    cursor.close()
                    conn.close()
                    final_statements.pop()
                    return final_statements
    cursor.close()
    conn.close()
    raise ValueError(f"Even with 0 rows, token count is too high!")

def clean_creates(sql_text: str) -> str:
    """While these fields might be useful for some purposes, I've honestly
    needed them so rarely as a data scientist that we are going to exclude them
    """

    def replace_(node: Expression) -> Optional[Expression]:
        if isinstance(
            node,
            (
                expressions.ColumnConstraint,
                expressions.PrimaryKey,
            ),
        ):
            return None
        return node

    return str(sqlglot.parse_one(sql_text).transform(replace_))

def hard_replace__clean_creates(sql_text: str):
    """The backticks and double-quotes are always equivalent in bird
    # but sqlglot cannot yet handle the backticks
    """
    try:
        return clean_creates(
            sql_text.replace("`", '"')
            .replace("WITHOUT ROWID", "")
            .replace("on update cascade", "")
            .replace("ON UPDATE CASCADE", "")
            .replace("on delete cascade", "")
            .replace("ON DELETE CASCADE", "")
            .replace("references staff", "")
        )  # .sql()
    except Exception:
        raise

def read_in_all_sqlite_dbs():
    """Read in all the sqlite databases from the bird data"""
    dirs = glob.glob(DB_DIR + "/*")
    statements = []
    for d in dirs:
        if os.path.isfile(d):
            continue
        dbname = d.split("/")[-1]
        sqlite_db_path = os.path.join(d, dbname + ".sqlite")
        assert os.path.exists(sqlite_db_path), f"DB {sqlite_db_path} does not exist!"
        ddl_list = get_create_table_and_data(sqlite_db_path)
        for ddl in ddl_list:
            statements.append((dbname, hard_replace__clean_creates(ddl)))

    return statements

def make_x(tables, db_id, ideal_sql):
    """Make the x and y for the training data"""
    return tables, db_id, ideal_sql

ddl_statements = read_in_all_sqlite_dbs()

df_ddl = pd.DataFrame(ddl_statements)
df_ddl.columns = ["db_id", "ddl"]
df_ddl = df_ddl.groupby("db_id")["ddl"].agg("\n".join).reset_index(name="ddl")

def format_ddl(ddl_str):
    formatted_ddls = []

    # Split the ddl_str by "CREATE TABLE"
    create_tables = re.split(r"(?i)CREATE TABLE", ddl_str)

    for ct in create_tables:
        if not ct.strip():
            continue

        # Extract table name from the current CREATE TABLE section
        table_name_match = re.search(r'^\s*("?[\w\s-]+"?|[\w\s-]+)', ct)

        table_name = (
            table_name_match.group(1).strip() if table_name_match else "Unknown Table"
        )

        splits = ct.split("INSERT INTO")

        columns = re.sub(r"^\s*" + re.escape(table_name), "", splits[0]).strip()
        if columns.startswith("(") and columns.endswith(")"):
            columns = columns[1:-1]

        columns = " ".join(columns.split())

        cleaned_table_name = table_name.strip('"')
        insert_statements = [
            split.replace(f"{cleaned_table_name} VALUES", "").strip()
            for split in splits[1:]
        ]

        insert_statements = [
            stmt.replace("(", "").replace(")", "") for stmt in insert_statements
        ]

        if "VARBINARY" in ct:
            formatted_ddl = table_name + " (" + columns + ");"
        else:
            formatted_ddl = (
                table_name
                + " ("
                + columns
                + ");\nINSERT INTO "
                + table_name
                + " VALUES\n("
                + ")\n(".join(insert_statements)
                + ");"
            )
        formatted_ddls.append(formatted_ddl)

    return "\n".join(formatted_ddls)

df_ddl["ddl"] = df_ddl["ddl"].apply(format_ddl)

df_question = pd.read_json(JSON_FILE)
joined_df = pd.merge(df_question, df_ddl, on=["db_id"])

df = pd.DataFrame(
    joined_df.apply(
        lambda x: make_x(
            x["ddl"]
            + "\n## The user has asked:\n"
            + x["question"]
            + "\nNOTE: "
            + x["evidence"],
            x["db_id"],
            x["SQL"],
        ),
        axis=1,
    ).tolist()
)
df.columns = ["user_prompt", "db_id", "ideal_assistant_response"]

df = df.drop_duplicates()
df['tokens'] = df.apply(lambda row: num_tokens_from_messages([{"role": "user", "content": row['user_prompt']}]), axis=1)

# Save the processed data to a JSONL file
file_name = "fine_tuning_bird_qna_take_two_training_data.jsonl"
with open(file_name, 'w') as outfile:
    for row in df.itertuples():
        json.dump({"messages": [{"role": "user", "content": row.user_prompt}, 
                                {"role": "assistant", "content": row.ideal_assistant_response}]}, outfile)
        outfile.write('\n')

# Upload the training file
training_file = openai.File.create(
    file=open(file_name),
    purpose='fine-tune'
)

# Start the fine-tuning job
fine_tune_response = openai.FineTune.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo",  # Specify the base model you want to fine-tune
    n_epochs=4  # Adjust the number of epochs as needed
)

print(f"Fine-tuning job started: {fine_tune_response['id']}")

# Monitor the fine-tuning job
import time

while True:
    fine_tuning_jobs = openai.FineTune.list()
    for job in fine_tuning_jobs['data']:
        print(f"Job ID: {job['id']}, Status: {job['status']}")
    if all(job['status'] in ['succeeded', 'failed'] for job in fine_tuning_jobs['data']):
        break
    time.sleep(60)  # Check every minute

# Print the result of the fine-tuning job
fine_tuning_jobs = openai.FineTune.list()
for job in fine_tuning_jobs['data']:
    print(f"Job ID: {job['id']}, Status: {job['status']}, Result: {job.get('fine_tuned_model')}")
