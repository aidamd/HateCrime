import json
import pandas

def getjson(texts, name):
    with open(name + ".json", "w") as fi:
        for i in range(len(texts)):
            entity = {
                "tid": str(i),
                "timestamp": 1542244186,
                "text": texts[i],
                "full_text": texts[i],
                "retweet": False,
                "entities": []
            }
            json.dump(entity, fi)
            fi.write("\n")

patch = pandas.read_csv("patch_sample.csv")
getjson(patch["text"].tolist(), "patch")
propublica = pandas.read_csv("propublica_sample.csv")
getjson(propublica["text"].tolist(), "propublica")