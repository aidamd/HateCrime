import pandas as pd
import glob
import math

dataset = {"state": [],
           "city": [],
           "year": [],
           "count": [],
           "statecity": []
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
        if row[1] == "Cities":
            agency = True
        elif isinstance(row[1], str):
            agency = False
        if agency:
            if not isinstance(row[2], str):
                continue
            else:
                city = row[2]
                try:
                    count = sum([row[3 + j] for j in range(cols[year])])
                except Exception:
                    print()
                state = state.replace("District of Columbia", "District-Columbia").replace(" ", "-")
                city = city.replace("New York", "New York City")
                dataset["state"].append(state)
                dataset["city"].append(city)
                dataset["year"].append(year)
                dataset["count"].append(count)
                dataset["statecity"].append(state.lower() + ", " + city.lower())
pd.DataFrame.from_dict(dataset).to_csv("FBI-hate-cities.csv", index=False)