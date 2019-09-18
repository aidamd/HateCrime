import pandas as pd


#patch = pd.read_csv("patch_500.tsv", delimiter="\t")
#propublica = pd.read_csv("propublica_500.tsv", delimiter="\t")

annotated_files = ["kidnap.tsv"]

text = list()
homicide = list()

for file in annotated_files:
    print(file)
    p = pd.read_csv("/home/aida/Projects/IE-final/Data/kidnap/" + file, delimiter="\t")
    print(p.shape)
    foundations = dict()
    texts = dict()
    for i, row in p.iterrows():
        try:
            foundations.setdefault(row["Tweet ID"], list()).extend(row["Foundation"].split(","))
        except Exception:
            print()
        texts[row["Tweet ID"]] = row["Text"]
    print(len(texts.keys()))

    for id in foundations.keys():
        anno = foundations[id]
        text.append(texts[id])
        if "kidnap" in anno:
            homicide.append(1)
        else:
            homicide.append(0)

train = pd.DataFrame.from_dict({"text": text,
                                "labels": homicide})
train.to_csv("/home/aida/Projects/IE-final/Data/kidnap/train_kidnap.csv")
