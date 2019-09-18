import pandas as pd
import glob
import math
import re

dataset = {"state": [],
           "county": [],
           "year": [],
           "count": [],
           "statecounty": []
           }

for file in glob.glob("hate/*"):
    df = pd.read_excel(file)
    year = int(file[5:-4])
    print(year)
    agency = False
    state = ""
    cols = {
        2012:5,
        2013:7,
        2014:7,
        2015:6,
        2016:6,
        2017:6
    }
    for i, row in df.iterrows():
        if isinstance(row[0], str):
            state = row[0]
        if row[1] == "Metropolitan Counties" or row[1] == "Nonmetropolitan Counties":
            agency = True
        elif isinstance(row[1], str):
            agency = False
        if agency:
            if not isinstance(row[2], str):
                continue
            else:
                county = row[2]
                try:
                    count = sum([row[3 + j] for j in range(cols[year])])
                except Exception:
                    print()
                state = state.lower()
                state = state.replace("District of Columbia", "District-Columbia").replace(" ", "-")
                county = re.sub(" (County)[a-zA-Z ]+", "", county)
                dataset["state"].append(state)
                dataset["county"].append(county)
                dataset["year"].append(year)
                dataset["count"].append(count)
                dataset["statecounty"].append(state.lower() + ", " + county.lower())
pd.DataFrame.from_dict(dataset).to_csv("FBI-hate-counties.csv", index=False)