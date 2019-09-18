import pandas as pd
import urllib
from bs4 import BeautifulSoup

"""
data = pd.read_csv("../Data/patch_more3.csv")
links = data["link"].tolist()
cities = dict()
city_col = list()

for link in links:
    patch = link.split("/")[3] + link.split("/")[4]
    if patch in cities.keys():
        continue
    if "across" in patch or "Across" in patch:
        cities[patch] = ""
        continue
    try:
        html = urllib.request.urlopen(link).read().decode()
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        continue
    #try:
    #    sub_links = soup.findAll("ul", {"class": "dropdown-menu sections-sub-menu"})
    #    sub_links = [s for sub in sub_links for s in sub.findChildren("a")]
    #    cities[patch] = [sub.text for sub in sub_links if sub["trigger"] == "town_info"][0]
    #except Exception:
    cities[patch] = soup.find("a", {"class": "logo-title text-center text-dark"}).text

print([j for j in cities.keys() if cities[j] == ""])

for i, row in data.iterrows():
    city_col.append(cities[row["state"] + row["patch"]])
data["city"] = pd.Series(city_col)
data.to_csv("../Data/patch_more3_city.csv", index = False)
"""

html = urllib.request.urlopen("https://patch.com/map").read().decode()
soup = BeautifulSoup(html, "html.parser")

cities = dict()

a = soup.findAll("a", {"trigger": "map_links_patch"})
for i in a:
    cities[i["href"].split("/")[-2] + "/" + i["href"].split("/")[-1]] = i.text

print(len(set(cities.values())))
data = pd.read_csv("../Data/patch_more3.csv")
city_col = list()
for i, row in data.iterrows():
    link = row["link"]
    city_col.append(cities[link.split("/")[3] + "/" + link.split("/")[4]])
data["city"] = pd.Series(city_col)
data.to_csv("../Data/patch_more3_city.csv", index = False)