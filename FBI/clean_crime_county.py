import pandas as pd
import glob
import math

dataset = {"state": [],
           "county": [],
           "year": [],
           "statecounty": [],
           "kidnap": [],
           "homicide": []
           }

for file in glob.glob("crime/*"):
    s = False
    df = pd.read_excel(file)
    year = int(file[6:-4])
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
        if s:
            homicide = [x for x in range(len(row)) if row[x] == "Homicide\nOffenses"][0]
            kidnap = [x for x in range(len(row)) if row[x] == "Kidnapping/\nAbduction"][0]
            s = False
        if row[0] == "State":
            s = True
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
                    homicides = row[homicide]
                    kidnaps = row[kidnap]
                except Exception:
                    print()
                state = state.replace("District of Columbia", "District-Columbia").replace(" ", "-")
                county = county.replace("New York", "New York City")
                dataset["state"].append(state)
                dataset["county"].append(county)
                dataset["year"].append(year)
                dataset["statecounty"].append(state.lower() + ", " + county.lower())
                try:
                    dataset["homicide"].append(int(homicides) if not math.isnan(homicides) else 0)
                except Exception:
                    print()
                dataset["kidnap"].append(int(kidnaps) if not math.isnan(kidnaps) else 0)
pd.DataFrame.from_dict(dataset).to_csv("FBI-crime-counties.csv", index=False)