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
import numpy as np
import scipy
import ast # convert string literal list into real list.
from sklearn.metrics.pairwise import cosine_similarity

### Google translate api credentials.
token = 'AIzaSyBZx4GANyssAEQdVlG2XuSeY-8vUsxRkBw'

### Initializa gensim word2vec model. Around 2~3 minutes.
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

### Given Chinese title, use Google translate api to make it into English.
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
# with open(new_ada_content_en_vec, 'w') as target:
#     fieldnames = ['id', 'title', 'tags' , 'wordvec', 'author', 'date', 'url', 'content']
#     writer = csv.DictWriter(target, fieldnames=fieldnames)
#     writer.writeheader()

#     with open(new_ada_content_en) as source:
# 	    reader = csv.DictReader(source.read().splitlines())
# 	    for row in reader:
	    	
# 	    	# [1] Text sanitization; then remove duplicate spaces into one.
# 	    	sani_title = regex.sub(' ', row['title'])
# 	    	sani_title = re.sub(' +',' ',sani_title).strip()

# 	    	# [2] NLTK tagging, grab the noun of the sentence.
# 	    	results = filter(lambda (a,b): b in ['NN', 'NNS', 'NNP', 'NNPS'] and len(a) > 3, pos_tag(word_tokenize(sani_title)))
	    	
# 	    	# [3] Given results, filter unimportant pair.
# 	    	# - keywords must more than three characters. 
# 	    	# - all lower case.
# 	    	# - if exist several "NNP", only get the first one.
# 	    	# 
# 	    	# Notice: tags list can be empty!
	    	
# 	    	def get_first_nnp(pair_lst):
# 	    		nnp_counter = 0
# 	    		new_lst = []
# 	    		for word, tag in pair_lst:
# 	    			if tag == 'NNP' and nnp_counter < 1:
# 	    				new_lst.append((word.lower(), tag))
# 	    				nnp_counter += 1
# 	    			elif tag in ['NN', 'NNS', 'NNPS']:
# 	    				new_lst.append((word.lower(), tag))
# 	    		return new_lst

# 	    	def get_vector(pair_lst):
# 	    		# pass through model, and sum up all vector.
# 	    		vec_lst = []
# 	    		if pair_lst:
# 		    		for word, tag in pair_lst:
# 		    			if word in model:
# 		    				vec_lst.append(model[word])
# 		    	else:
# 		    		return []
# 		    	if vec_lst:
# 		    		sum_vec = [sum(column) for column in zip(*vec_lst)]
# 		    	else:
# 		    		# in case vec_lst is empty, meaning word not in model like utevents.
# 		    		return []
# 	    		return sum_vec

# 	    	results = get_first_nnp(results)
# 	    	word_vector = get_vector(results)
# 	    	# print sani_title, results, word_vector
# 	    	writer.writerow({'id': row['id'], 'title': sani_title, 'tags': results, 'wordvec': word_vector, 'author': row['author'], 'date': row['date'], 'url': row['url'], 'content': row['content']})
# 	    	print 'Processing vec NO. ' + str(row['id'])


### Given vector, user inputs a word and find the most similar match.
example_vec = [ 0.15234375,  0.00878906, -0.13769531,  0.1640625 ,  0.11523438,
       -0.02026367, -0.16503906, -0.00297546,  0.20214844,  0.05004883,
        0.42773438, -0.20898438,  0.19238281, -0.03637695, -0.01916504,
       -0.16113281,  0.19140625,  0.37109375,  0.05029297,  0.10595703,
        0.12988281, -0.38476562,  0.21679688,  0.15917969, -0.03881836,
        0.09423828, -0.07470703,  0.03735352,  0.29101562, -0.31835938,
       -0.03173828, -0.11523438,  0.11132812,  0.18554688, -0.11230469,
       -0.27734375, -0.18457031,  0.16796875, -0.45898438, -0.07763672,
       -0.36523438,  0.0267334 ,  0.3046875 , -0.22265625, -0.28710938,
       -0.46875   ,  0.03173828,  0.265625  ,  0.23339844, -0.078125  ,
       -0.1484375 , -0.08105469,  0.02502441, -0.09179688,  0.11865234,
       -0.25195312,  0.06787109,  0.23242188,  0.00230408,  0.11816406,
        0.19238281, -0.01312256,  0.04785156, -0.18945312, -0.02124023,
        0.03442383, -0.02856445, -0.01074219,  0.21289062,  0.33203125,
        0.16699219,  0.15332031,  0.19335938,  0.02038574, -0.8203125 ,
       -0.04638672, -0.140625  , -0.15136719, -0.04052734, -0.3203125 ,
        0.13671875,  0.07275391, -0.11181641, -0.17480469, -0.19726562,
       -0.34960938,  0.04785156,  0.04248047, -0.08886719,  0.30664062,
       -0.10498047, -0.04296875,  0.04443359, -0.10205078,  0.32226562,
       -0.4609375 , -0.10302734, -0.16113281,  0.18261719,  0.03735352,
       -0.05151367, -0.01586914,  0.35742188,  0.2890625 ,  0.1953125 ,
       -0.00872803,  0.45507812,  0.03979492,  0.1875    , -0.13769531,
       -0.01049805, -0.06787109,  0.1171875 ,  0.10449219, -0.26367188,
        0.2890625 ,  0.20019531,  0.08007812,  0.24414062,  0.10498047,
       -0.27539062,  0.08691406, -0.20898438, -0.27539062,  0.06298828,
        0.13378906,  0.00909424, -0.10742188,  0.05517578,  0.17089844,
       -0.10400391, -0.0480957 , -0.53125   ,  0.07861328, -0.01055908,
        0.05004883,  0.21582031,  0.21484375, -0.06884766,  0.10302734,
        0.10253906, -0.28710938, -0.06787109, -0.04663086,  0.12402344,
       -0.14550781, -0.11279297,  0.09472656, -0.00738525,  0.11425781,
        0.25      , -0.08935547,  0.08496094, -0.05175781, -0.17578125,
       -0.20605469, -0.43945312, -0.15039062, -0.23730469,  0.12792969,
       -0.27539062, -0.15234375,  0.49804688,  0.04003906,  0.13378906,
       -0.08300781,  0.07080078, -0.23046875, -0.19238281,  0.00747681,
        0.14746094,  0.22753906, -0.12402344,  0.20605469, -0.08056641,
        0.0189209 ,  0.29296875, -0.23730469, -0.0625    , -0.12695312,
       -0.03857422, -0.125     , -0.0625    , -0.05273438,  0.20800781,
        0.11132812, -0.01696777, -0.00793457, -0.1875    ,  0.24414062,
        0.10986328,  0.19433594, -0.0703125 ,  0.11767578, -0.01708984,
        0.57421875, -0.16894531, -0.20703125, -0.28125   , -0.26757812,
       -0.08398438,  0.15039062, -0.28320312, -0.07324219, -0.17871094,
       -0.06054688,  0.00346375,  0.37695312,  0.2890625 ,  0.21582031,
        0.28710938,  0.0234375 , -0.35742188,  0.15234375, -0.24316406,
        0.38476562, -0.03271484,  0.73828125, -0.29882812, -0.13085938,
       -0.18457031, -0.10791016, -0.0189209 ,  0.203125  , -0.12304688,
       -0.15234375, -0.375     , -0.12792969, -0.3125    , -0.10791016,
        0.03222656,  0.31445312, -0.58984375,  0.18164062,  0.11474609,
        0.11035156, -0.00531006,  0.18945312,  0.2890625 , -0.22460938,
       -0.07861328,  0.04882812, -0.0534668 , -0.28125   , -0.09667969,
        0.08837891,  0.08203125, -0.2578125 ,  0.09765625, -0.14453125,
        0.14550781, -0.05078125, -0.04321289,  0.21875   , -0.09570312,
        0.24414062,  0.09521484,  0.01300049,  0.11914062, -0.13671875,
        0.23535156,  0.21191406, -0.10595703,  0.04248047, -0.20117188,
       -0.05883789, -0.02880859, -0.25585938, -0.44335938,  0.20996094,
        0.02490234,  0.11279297,  0.03833008,  0.39257812, -0.06689453,
       -0.09423828, -0.02416992,  0.07568359,  0.03613281, -0.09570312,
        0.3203125 ,  0.04541016,  0.31054688, -0.33203125,  0.1953125 ,
        0.31835938, -0.359375  ,  0.43359375, -0.11767578,  0.05834961,
       -0.16113281, -0.06787109,  0.28320312,  0.09130859,  0.19433594,
        0.08007812, -0.18066406,  0.16015625, -0.08886719, -0.23828125]
example_words = ['luggage', 'food', 'professor', 'pizza', 'park', 'autumn', 'party', 'music', 'festival', 'passport', 'toronto', 'japanese', 'rank', 'house', 'mathematics', 'chrismas', 'engineering', 'helloween']
allvec_lst = []
vec_dimension = len(example_vec)
null_vec = [0.0001] * vec_dimension
title_lst = []
with open(new_ada_content_en_vec) as source:
	reader = csv.DictReader(source.read().splitlines())
	for row in reader:		
		# notice row['wordvec'] returns a string, not a list.
		real_vec = ast.literal_eval(row['wordvec'])
		if real_vec: # can't do this, since we need the index to trace back the right entry!!
			allvec_lst.append(real_vec)
		else:
			allvec_lst.append(null_vec)

with open(new_ada_content) as source:
	reader = csv.DictReader(source.read().splitlines())
	for row in reader:		
		title_lst.append(row['title'].strip().decode('utf-8'))



# def cos_cdist(matrix, vector):
#     """
#     Compute the cosine distances between each row of matrix and vector.
#     """
#     v = vector.reshape(1, -1)
#     return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)

def cos_similarity(nested_lst, lst):
	results = map(lambda x: cosine_similarity(x, lst)[0][0], nested_lst)
	return results

print_result = []
for test_word in example_words:
	if test_word in model:
		test_vec = model[test_word]
		dist = cos_similarity(allvec_lst, test_vec)
		max_id = dist.index(max(dist))
		print_result.append([test_word, title_lst[max_id]])
	else:
		test_vec = []
		print test_word, " don't have a matching vector."

for pair in print_result:
	print pair[0], pair[1]

# matrix = np.array(allvec_lst)
# test_vec = np.array(example_vec)
# dist = cos_cdist(matrix, test_vec)
# dist = cos_similarity(allvec_lst, example_vec)
# print dist
# find the largest number inside the distance matrix.
# max_id = dist.argmax()
# max_id = dist.index(max(dist))
# print max_id, dist[max_id]












