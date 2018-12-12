import pandas as pd

df = pd.read_csv("train.csv")

targets, acts = [], []

race = ["jew", "black", "african", "asian"]
nation = ["indian", "chinese"]
gender = ["queer", "trans", "lgbt"]
religion = ["muslim", "shikh", "mosque"]
sex = ["gay", "lesbian"]

assault = ["stab", "murder", "attack", "punch"]
arson = ["fire"]
vand = ["vandal", "graffit"]
demo = ["slur"]


count = 0
for idx, row in df.iterrows():
    if row["labels"] == 1:
        target = []
        act = []
        text = row["text"].lower()
        for i in race:
            if i in text:
                target.append(1)
                break
        for i in nation:
            if i in text:
                target.append(2)
                break
        for i in gender:
            if i in text:
                target.append(3)
                break
        for i in religion:
            if i in text:
                target.append(4)
                break
        for i in sex:
            if i in text:
                target.append(5)
                break
        if len(target) > 0:
            targets.append(target[0])
        else:
            targets.append(0)

        for i in assault:
            if i in text:
                act.append(1)
                break
        for i in arson:
            if i in text:
                act.append(2)
                break
        for i in vand:
            if i in text:
                act.append(3)
                break
        for i in demo:
            if i in text:
                act.append(4)
                break
        if len(act) > 0:
            acts.append(act[0])
        else:
            acts.append(0)
    else:
        targets.append(6)
        acts.append(5)

df["target"] = targets
df["action"] = acts
df.to_csv("train_anno.csv")