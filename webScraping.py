# https://www.digitalocean.com/community/tutorials/how-to-scrape-web-pages-with-beautiful-soup-and-python-3
#https://www.scrapehero.com/web-scraping-tutorial-for-beginners-part-3-navigating-and-extracting-data/#putting-it-all-together-into-a-function14
import requests
from bs4 import BeautifulSoup

text_file = open("scrap.txt","w")

def parse_description(url):
    full_description = url
    page = requests.get(full_description)
    soup = BeautifulSoup(page.text, 'html.parser')
    container = soup.find(class_='col-12 col-md-8')
    p_tag = container.find('p')
    descrip = p_tag.contents[0]

    #originally had it printing, but now writing to text file
    # https://stackoverflow.com/questions/37412377/write-data-scrapped-to-text-file-with-python-script
    text_file.write(descrip)

def traverse_page(url):
    new_page = url
    page = requests.get(new_page)
    soup = BeautifulSoup(page.text, 'html.parser')
    rows = soup.find_all(class_='row-link')

    a_s = []
    for a_tag in rows:
        a_s.append(a_tag.find('a'))

    urls = []
    for a_tag in a_s:
        url = a_tag['href']
        if not url.startswith('http'):
            url = "https://harassmap.org/en/reports"+url
        urls.append(url)

    for item in urls:
        parse_description(item)

quote_page = 'https://harassmap.org/en/reports'
page = requests.get(quote_page)
soup = BeautifulSoup(page.text, 'html.parser')

container=soup.find(class_='pagination')
a_tag = container.find_all('a')

urls = []
for a_s in a_tag:
    url = a_s['href']
    urls.append(url)
for item in urls:
    traverse_page(item)





# quote_page = 'https://harassmap.org/en/reports'
# page = requests.get(quote_page)
# soup = BeautifulSoup(page.text, 'html.parser')
# rows = soup.find_all(class_='row-link')
#
#
# a_s = []
# for a_tag in rows:
#     a_s.append(a_tag.find('a'))
#
#
# urls = []
# for a_tag in a_s:
#     url = a_tag['href']
#     if not url.startswith('http'):
#         url = "https://harassmap.org/en/reports"+url
#     urls.append(url)
#
# # def traverse_page(url):
# #     new_page = url
# #     page = requests.get(new_page)
# #     parse_description()
#
# def parse_description(url):
#     full_description = url
#     page = requests.get(full_description)
#     soup = BeautifulSoup(page.text, 'html.parser')
#     container = soup.find(class_='col-12 col-md-8')
#     p_tag = container.find('p')
#     descrip = p_tag.contents[0]
#
#     print(descrip)
#
# for item in urls:
#     parse_description(item)
