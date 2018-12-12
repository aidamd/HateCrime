# patch.com news scraper
# written by Jun Yen Leung for CSSL (USC)

###### IMPORTS ######
from bs4 import BeautifulSoup
import mysql.connector
import urllib.request
from datetime import datetime


###### SETTINGS ######
SQL_CONFIG = {
	'user':'root',
	'password':'ynhkatda',
	'host':'localhost',
	'database':'Patch',
	'raise_on_warnings':True,
	'charset':'utf8mb4',
	'collation':'utf8mb4_bin'
}
CATEGORIES = ["police-fire"]


###### GLOBALS ######
PATCH_LINKS = []
SQL_CONNECTION = ""
SQL_CURSOR = ""


###### METHODS ######
# make a string sql safe
def makeSafe(string):
	return string.replace("'", "").replace('"', "").replace("`", "")

# initialize connections
def init():
	global SQL_CONNECTION
	global SQL_CURSOR
	SQL_CONNECTION = mysql.connector.connect(**SQL_CONFIG)
	SQL_CURSOR = SQL_CONNECTION.cursor(buffered=True)

# get patches to scrape
def getPatches():
	global PATCH_LINKS
	try:
		# get patch raw data
		html = urllib.request.urlopen("https://patch.com/map").read().decode()
		soup = BeautifulSoup(html, "html.parser")
		# parse with bs4
		patch_links = soup.findAll("a", {"category" : "nav_primary"})
		for patch_link in patch_links[9:]:
			patch_href = patch_link["href"]
			# get rid of stray links
			if patch_href.count("/") <= 3:
				continue
			PATCH_LINKS.append(patch_href)
		return len(PATCH_LINKS)
	except Exception as e:
		error = "FAILED to get patches: {}".format(e)
		
# get news for a particular patch link
def getNews(category, patch_link):
	try:
		# get state and patch data
		state_patch = patch_link[18:].split("/")
		state = state_patch[0]
		patch = state_patch[1]
		# prepare to loop
		base_link = patch_link + "/" + category + "?page="
		done = False
		page = 1
		# check pages until no articles remain
		while(done == False):
			print("PAGE: {}".format(page))
			# get page of articles
			url = base_link + str(page)
			html = urllib.request.urlopen(url).read().decode()
			# parse page of articles
			soup = BeautifulSoup(html, "html.parser")
			article_footers = soup.findAll("div", {"class" : "slot-footer"})
			num_articles = len(article_footers)
			# there are more near-black-links than there are articles, so limit using num_articles
			article_headers = soup.findAll("a", {"class" : "near-black-link"})[:num_articles]
			# for each article...
			for i, article_header in enumerate(article_headers):
				try:
					# get header data
					article_title = article_header["title"]
					#print(article_title)
					article_href = article_header["href"]
					# get author
					article_footer = article_footers[i].text.strip().split(",")
					author = article_footer[0][3:]
					# correct links
					if article_href[:17] != "https://patch.com":
						article_href = "https://patch.com" + article_href
					# get article from link
					article_html = urllib.request.urlopen(article_href).read().decode()
					article_soup = BeautifulSoup(article_html, "html.parser")
					article_wrapper = article_soup.find("div", {"id" : "article-wrapper"})
					article_text_ps = article_wrapper.findAll("p")
					article_text = ""
					for p in article_text_ps:
						article_text += p.text.strip() + " "
					article_text = article_text.strip()
					# get date
					date = article_soup.find("time")["datetime"].split()[0]
					statement = """INSERT INTO Patch.Articles (author, state, patch, category, date, title, link, text) VALUES ("{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}")""".format(makeSafe(author), state, patch, category, date, makeSafe(article_title), article_href, makeSafe(article_text))
					# print(statement)
					SQL_CURSOR.execute(statement)
					SQL_CONNECTION.commit()
				except Exception as e:
					error = "FAILED to get an article: {}".format(e)
					print(error)
					continue
			page += 1
			# no more articles -> done
			if num_articles == 0:
				print("DONE")
				done = True
	except Exception as e:
		error = "FAILED to get news for patch {}: {}".format(patch_link, e)
			
###### MAIN SEQUENCE ######
START_TIME = datetime.now()
init()
print("IDENTIFIED {} PATCHES".format(getPatches()))
for category in CATEGORIES:
	for patch_link in PATCH_LINKS:
		print("--------------------")
		print("CATEGORY: {}".format(category))
		print("PATCH: {}".format(patch_link))
		getNews(category, patch_link)
END_TIME = datetime.now()
print("--------------------")
print("START TIME: {}".format(START_TIME))
print("END TIME: {}".format(END_TIME))