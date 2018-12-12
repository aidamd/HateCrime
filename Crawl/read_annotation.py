import pandas as pd


patch = pd.read_csv("patch_500.tsv", delimiter="\t")
propublica = pd.read_csv("propublica_500.tsv", delimiter="\t")

text = list()
target = list()
action = list()
hate = list()


for i, row in patch.iterrows():
    anno = row["Foundation"]
    text.append(row["Text"])
    if "nhc" in anno:
        hate.append(0)
        target.append(8)
        action.append(5)
    else:
        hate.append(1)
        if "rae" in anno:
            target.append(0)
        elif "nat" in anno:
            target.append(1)
        elif "gen" in anno:
            target.append(2)
        elif "rel" in anno:
            target.append(3)
        elif "sxo" in anno:
            target.append(4)
        elif "idl" in anno:
            target.append(5)
        elif "pol" in anno:
            target.append(6)
        elif "mph" in anno:
            target.append(7)
        else:
            print(row["Text"])
        if "assault" in anno:
            action.append(0)
        elif "arson" in anno:
            action.append(1)
        elif "van" in anno:
            action.append(2)
        elif "demo" in anno:
            action.append(3)
        else:
            print(row["Text"])

train = pd.DataFrame.from_dict({"text": text,
                                "labels": hate,
                                "target": target,
                                "action": action})
train.to_csv("train_patch.csv")