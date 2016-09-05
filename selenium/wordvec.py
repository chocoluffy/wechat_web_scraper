# -*- coding: utf-8 -*-
import urllib
from bs4 import BeautifulSoup
import re
import csv
import requests
import json
import re
from nltk import word_tokenize, pos_tag
import gensim
from timeit import default_timer as timer

### Google translate api credentials.
token = 'AIzaSyBZx4GANyssAEQdVlG2XuSeY-8vUsxRkBw'

### Initializa gensim word2vec model.
start = timer()
model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
end = timer()
print 'Loading word2vec model takes: ', (end - start)

### Return compressed nested texts from page.
def url2content(url):
	html = urllib.urlopen(url).read()
	soup = BeautifulSoup(html, 'html.parser')
	texts = soup.findAll(text=True)

	def visible(element):
	    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
	        return False
	    elif re.match('<!--.*-->', element.encode('utf-8')):
	        return False
	    return True

	visible_texts = filter(visible, texts)

	# print visible_texts

	def removeSpace(string):
		return string.replace("\n", "")

	# for visible_text in map(removeSpace, visible_texts):
		# print visible_text.encode('utf-8')
	
	### From manually experiments, I find that the [0] refers to an link script tag, [1] refers to article title. [8] refers to the date, and [9] refers to the author.
	# for i in range(1, 10):
	# 	print filter(bool, map(removeSpace, visible_texts))[i].encode('utf-8')
	# bool filter is to remove empty ssrting from python list.
	sanitized_texts = filter(bool, map(removeSpace, visible_texts))
	title = sanitized_texts[1].encode('utf-8')
	date = sanitized_texts[8].encode('utf-8')
	author = sanitized_texts[9].encode('utf-8')
	combined_string = ''.join(sanitized_texts[1:]).encode('utf-8')

	# print combined_string # it looks good!
	# return combined_string.encode('utf-8')
	return {'title': title,
			'date': date,
			'author': author,
			'combined_string': combined_string
			}

## Define old stamp and new stamp followed by file name. Afterwards, any changes should only change this new_stamp, other parts should stay unchanged.
new_stamp = '160904'

base_dir = './data/'
new_ada = base_dir + 'ada' + new_stamp + '.csv'
new_ada_content = base_dir + 'wordvec' + new_stamp + '.csv'
new_ada_content_en = base_dir + 'wordvec-en' + new_stamp + '.csv'
new_ada_content_en_vec = base_dir + 'wordvec-en-vec' + new_stamp + '.csv'

##  Connect with ada.csv to port from url to texts
# with open(new_ada_content, 'w') as target:
#     fieldnames = ['id', 'title', 'author', 'date', 'url', 'content']
#     writer = csv.DictWriter(target, fieldnames=fieldnames)
#     writer.writeheader()

#     with open(new_ada) as source:
# 	    reader = csv.DictReader(source)
# 	    for row in reader:
# 	    	combo = url2content(row['url'])
# 	    	writer.writerow({'id': row['id'], 'title': combo['title'], 'author': combo['author'], 'date': combo['date'], 'url': row['url'], 'content': combo['combined_string']})
# 	    	print 'Processing scraper NO.' + str(row['id'])


# with open(new_ada_content_en, 'w') as target:
#     fieldnames = ['id', 'title', 'author', 'date', 'url', 'content']
#     writer = csv.DictWriter(target, fieldnames=fieldnames)
#     writer.writeheader()

#     with open(new_ada_content) as source:
# 	    reader = csv.DictReader(source.read().splitlines())
# 	    for row in reader:
# 			query = 'https://www.googleapis.com/language/translate/v2?key=' + token + '&source=zh&target=en&q=' + row['title'].strip()
# 			response = requests.get(query)
# 			data = response.json()
# 			en_title = data['data']['translations'][0]['translatedText'].encode('utf-8')
# 			print en_title
# 			writer.writerow({'id': row['id'], 'title': en_title, 'author': row['author'], 'date': row['date'], 'url': row['url'], 'content': row['content']})
# 			print 'Processing translator NO. ' + str(row['id'])

regex = re.compile('[^a-zA-Z]')

### Given translated English title, [1] text sanitization [2] NLTK tagging [3] Word2Vec inferrence
with open(new_ada_content_en_vec, 'w') as target:
    fieldnames = ['id', 'title', 'tags' , 'wordvec', 'author', 'date', 'url', 'content']
    writer = csv.DictWriter(target, fieldnames=fieldnames)
    writer.writeheader()

    with open(new_ada_content_en) as source:
	    reader = csv.DictReader(source.read().splitlines())
	    for row in reader:
	    	
	    	# [1] Text sanitization; then remove duplicate spaces into one.
	    	sani_title = regex.sub(' ', row['title'])
	    	sani_title = re.sub(' +',' ',sani_title).strip()

	    	# [2] NLTK tagging, grab the noun of the sentence.
	    	results = filter(lambda (a,b): b in ['NN', 'NNS', 'NNP', 'NNPS'] and len(a) > 3, pos_tag(word_tokenize(sani_title)))
	    	
	    	# [3] Given results, filter unimportant pair.
	    	# - keywords must more than three characters. 
	    	# - all lower case.
	    	# - if exist several "NNP", only get the first one.
	    	# 
	    	# Notice: tags list can be empty!
	    	
	    	def get_first_nnp(pair_lst):
	    		nnp_counter = 0
	    		new_lst = []
	    		for word, tag in pair_lst:
	    			if tag == 'NNP' and nnp_counter < 1:
	    				new_lst.append((word.lower(), tag))
	    				nnp_counter += 1
	    			elif tag in ['NN', 'NNS', 'NNPS']:
	    				new_lst.append((word.lower(), tag))
	    		return new_lst

	    	def get_vector(pair_lst):
	    		# pass through model, and sum up all vector.
	    		vec_lst = []
	    		if pair_lst:
		    		for word, tag in pair_lst:
		    			if word in model:
		    				vec_lst.append(model[word])
		    	else:
		    		return []
		    	if vec_lst:
		    		sum_vec = [sum(column) for column in zip(*vec_lst)]
		    	else:
		    		# in case vec_lst is empty, meaning word not in model like utevents.
		    		return []
	    		return sum_vec

	    	results = get_first_nnp(results)
	    	word_vector = get_vector(results)
	    	# print sani_title, results, word_vector
	    	writer.writerow({'id': row['id'], 'title': sani_title, 'tags': results, 'wordvec': word_vector, 'author': row['author'], 'date': row['date'], 'url': row['url'], 'content': row['content']})
	    	print 'Processing vec NO. ' + str(row['id'])















