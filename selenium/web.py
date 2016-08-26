### use selenium to simulate scrollDown event.
import time
import re
import csv
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# UPDATE: 16.08.25, 120 articles.
urls = ["https://mp.weixin.qq.com/mp/homepage?__biz=MjM5MjAyOTEzMg==&hid=1&sn=5f0278051f4e3c1d2b774abfeff832f0&uin=MTExMzU3NjU0MA%3D%3D&key=cf237d7ae24775e83de30cd1bfdbc313e1de12905268c2fe19c1885b7280b7f882b26d854a2337282967ea7ba745c4b8&devicetype=iPhone+OS9.3.4&version=16031712&lang=en&nettype=3G+&fontScale=100&scene=1&from=groupmessage&isappinstalled=0",
	"http://mp.weixin.qq.com/mp/homepage?__biz=MjM5MjAyOTEzMg==&hid=1&sn=5f0278051f4e3c1d2b774abfeff832f0"]

browser = webdriver.Chrome()

browser.get(urls[0])
time.sleep(1)

def scrollDownAndConcatStr(concatString):
	elem = browser.find_element_by_tag_name("body")

	no_of_pagedowns = 20

	while no_of_pagedowns:
	    elem.send_keys(Keys.PAGE_DOWN)
	    time.sleep(0.2)
	    no_of_pagedowns-=1

	concatString = concatString + browser.page_source.encode('utf-8') # indeed print out the page html code.
	return concatString

articleString = ""
tabs = []
tabs = browser.find_elements_by_class_name("item")
articleString = scrollDownAndConcatStr(articleString)

# since there are 6 element that contains item as their class, the first three are the banner, we only need the last three elements.
# UPDATE: 16.08.25, in this version, there are four columns, tabs[4~6], apart from the front page.
for i in range(4, len(tabs)):
	tabs[i].click()
	articleString = scrollDownAndConcatStr(articleString)

# tabs[4].click() # will jump to the middle one.
# articleString = scrollDownAndConcatStr(articleString)

# tabs[5].click() # will jump to last panel.
# articleString = scrollDownAndConcatStr(articleString)

print articleString

postArr = []
postArr = re.findall(r'href="(http:\/\/mp\..*?)"', articleString)
print 'Grabbing ' + str(len(postArr)) + 'urls from website...'

### Remove duplicates from python list.
postArr = list(set(postArr))
print postArr
print 'After duplicates removal, ' + str(len(postArr)) + 'urls remained...'

with open('ada160825.csv', 'w') as csvfile:
    fieldnames = ['id', 'url']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for i in range(len(postArr)):
    	nosym_url = re.sub(r'&amp;', '&', postArr[i])
    	writer.writerow({'id': i, 'url': nosym_url})






















