import cssutils, time, queue, json
from bs4 import BeautifulSoup
from splinter import Browser
import newspaper
import mysql.connector
cnx = ""
cursor = ""
browser = ""

def make_connection():
    config = {
        'user': "root",
        'password': 'ynhkatda',
        'host': 'localhost',
        'database': 'Propublica',
        'raise_on_warnings': True,
        'charset': 'utf8mb4',
        'collation': 'utf8mb4_bin'
    }
    global cnx
    global cursor
    cnx = mysql.connector.connect(**config)
    cursor = cnx.cursor(buffered=True)

def getarticle(link):
    try:
        article = newspaper.Article(link)
        article.download()
        article.parse()

        author = ", ".join(a for a in article.authors)
        text = article.text
        article.nlp()
        keywords = ", ".join(a for a in article.keywords)
        summary = article.summary

        return (author, text, keywords, summary)
    except Exception:
        return None



def scrape():
    global browser, cursor, cnx
    # insert your own path-to-driver here
    executable_path = {'executable_path': '/home/aida/bin/chromedriver'}
    browser = Browser('chrome', **executable_path)

    browser.visit("https://projects.propublica.org/hate-news-index/")

    time.sleep(10)
    soup = BeautifulSoup(browser.html)
    date = ""
    articles = soup.findAll("div", {"class": "article"})
    print(len(articles))
    for article in articles:
        date_text = article.find("div", {"class": "article-date"}).text
        if date_text != "":
            try:
                date = time.strftime("%Y-%m-%d" , time.strptime(date_text.replace(".", ""), "%b %d, %Y"))
            except ValueError:
                date = time.strftime("%Y-%m-%d" , time.strptime(date_text.replace(".", ""), "%B %d, %Y"))
        link = article.find("a")["href"]
        source = article.find("span").text

        info = getarticle(link)

        if info:
            (author, text, keywords, summary) = info
        else:
            continue

        title = article.find("div", {"class": "article-title"}).text

        statement = """INSERT INTO Propublica.Articles (author, date, keywords, link, source, summary, text, title) VALUES ("{}", "{}", "{}", "{}", "{}", "{}", "{}", "{}")""".format(
            makeSafe(author), date, makeSafe(keywords), link, source, makeSafe(summary), makeSafe(text), makeSafe(title))
        try:
            cursor.execute(statement)
            cnx.commit()
        except Exception:
            continue

def makeSafe(string):
    return string.replace("'", "").replace('"', "").replace("`", "")


if __name__ == "__main__":
    make_connection()
    scrape()