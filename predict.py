import json
import pickle
import pandas as pd

params = json.load(open("params.json", "r"))

hate_batches = pickle.load(open("Data/" + params["dataset"] + "/predict.pkl", "rb"))
idx = [h[-1] for h in hate_batches]

patch = pd.read_csv("Data/patch_more3_city.csv")
hate = patch.iloc[idx, :]
print(hate.shape)
hate = hate.drop(columns = ["text", "title", "link"])
hate.to_csv("Data/" + params["dataset"] + "/" + params["dataset"] + "_articles.csv")