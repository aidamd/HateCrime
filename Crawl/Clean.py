import mysql.connector
import  pandas as pd
import re

def readPatch():
    config = {
        'user': "root",
        'password': 'ynhkatda',
        'host': 'localhost',
        'database': 'Patch',
        'raise_on_warnings': True,
        'charset': 'utf8mb4',
        'collation': 'utf8mb4_bin'
    }


    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor(buffered=True)

    df = pd.read_sql("select * from Articles", con = cnx)
    print("Number of documents in Patch: ", df.shape[0])
    """
    text = ['content', 'author', 'title', 'summary']
    for i, row in df.iterrows():
        for col in text:
            if row[col]:
                df.at[i, col] = row[col].decode("utf-8").lower()
    """
    df = df.drop_duplicates(subset = ['title'])
    print("Number of documents in Patch after removing duplicates: ", df.shape[0])
    for idx, row in df.iterrows():
        df.at[idx, "text"] = re.sub(r"([A-Z]| )+, ([A-Z][A-Z]) [-|—|–]", "", row["text"]).lstrip()
    df.to_csv("patch.csv")

def readPropublica():
    config = {
        'user': "root",
        'password': 'ynhkatda',
        'host': 'localhost',
        'database': 'Propublica',
        'raise_on_warnings': True,
        'charset': 'utf8mb4',
        'collation': 'utf8mb4_bin'
    }

    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor(buffered=True)

    df = pd.read_sql("select * from Articles", con=cnx)
    print("Number of documents in Propublica: ", df.shape[0])
    for idx, row in df.iterrows():
        row["title"].replace(row["source"], "")
    df.to_csv("propublica.csv")


readPatch()

