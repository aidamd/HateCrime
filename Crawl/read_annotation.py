import pandas as pd


#patch = pd.read_csv("patch_500.tsv", delimiter="\t")
#propublica = pd.read_csv("propublica_500.tsv", delimiter="\t")

annotated_files = ["patch3.tsv"]

text = list()
target = list()
action = list()
hate = list()

for file in annotated_files:
    print(file)
    p = pd.read_csv("/home/aida/Projects/IE-final/Data/" + file, delimiter="\t")
    print(p.shape)
    foundations = dict()
    texts = dict()
    for i, row in p.iterrows():
        foundations.setdefault(row["Tweet ID"], list()).extend(row["Foundation"].split(","))
        texts[row["Tweet ID"]] = row["Text"]
    print(len(texts.keys()))

    for id in foundations.keys():
        anno = foundations[id]
        text.append(texts[id])
        if "nhc" not in anno:
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
                #print(id)
                target.append(8)

            if "assault" in anno:
                action.append(0)
            elif "arson" in anno:
                action.append(1)
            elif "van" in anno:
                action.append(2)
            elif "demo" in anno:
                action.append(3)
            else:
                #print(id)
                action.append(5)
        else:
            hate.append(0)
            target.append(8)
            action.append(5)

train = pd.DataFrame.from_dict({"text": text,
                                "labels": hate,
                                "target": target,
                                "action": action})
train.to_csv("/home/aida/Projects/IE-final/Data/train_all.csv")
