import json
import pandas

keys = ["swastika", "hate", "racial", "religious", "religion", "gay", "transgender", "transsexual"]
pro = pandas.read_csv("../Data/propublica.csv")
patch = pandas.read_csv("../Data/patch_clean.csv")
count = 1
with open("patch2.json", "w") as fi:
    for _, row in pro.iterrows():
        if row["source"] == " Patch.com":
            entity = {
                "tid": count,
                "timestamp": 1542244186,
                "text": row["text"],
                "full_text": row["text"],
                "retweet": False,
                "entities": []
            }
            json.dump(entity, fi)
            fi.write("\n")
            count += 1
    for _, row in patch.iterrows():
        words = row["text"].lower().split()
        for word in keys:
            if word in words:
                entity = {
                    "tid": count,
                    "timestamp": 1542244186,
                    "text": row["text"],
                    "full_text": row["text"],
                    "retweet": False,
                    "entities": []
                }
                json.dump(entity, fi)
                fi.write("\n")
                count += 1
                break
print(count)
