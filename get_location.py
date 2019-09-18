import pandas as pd
from datetime import datetime
from pycorenlp import StanfordCoreNLP
"""
nlp_stan = StanfordCoreNLP('http://localhost:9900')

unreported = pd.read_csv("Data/hate/annotated_unreported.csv")
unreported = unreported.dropna(subset=["text"])


locations = list()
for i, row in unreported.iterrows():
    doc = row["text"]
    output = nlp_stan.annotate(doc, properties={
                                    'pipelineLanguage': 'en',
                                    'annotators': 'ner',
                                    'outputFormat': 'json'})
    doc_ner = list()
    for sent in output["sentences"]:
        for item in sent["tokens"]:
            word, ner = item['word'], item["ner"]
            if ner in ["CITY", "STATE_OR_PROVINCE"]:
                doc_ner.append(word)
    locations.append(", ".join([a for a in set(doc_ner)]))

unreported["locations"] = pd.Series(locations)

# Removing duplicated events
dup = list()
FMT = "%m/%d/%y"
for i, row in unreported.iterrows():
    if row["hc?"] == "n":
        dup.append(i)
    for j, rowj in unreported.iterrows():
        if j > i:
            if rowj["patch"] == row["patch"] and rowj["target"] == row["target"] and row["action"] == rowj["action"] and \
                abs(datetime.strptime(row["date"], FMT) - datetime.strptime(rowj["date"], FMT)).days < 2:
                dup.append(j)

unreported = unreported.drop(dup)

unreported.to_csv("annotated_unreported_cleaned.csv", index=False)
"""
unreported = pd.read_csv("annotated_unreported_cleaned.csv")

n = list()
for i, row in unreported.iterrows():
    if row["hc county matches newspaper county?"] == "n":
        for c in row["hc location"].split():
            found = False
            if isinstance(row["locations"], str) and c in row["locations"].lower().split():
                found = True
                break
        if not found:
            n.append(i)
print(n)