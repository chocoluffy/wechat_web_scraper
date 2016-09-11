# Update tag\wordvec based on literal_title
import csv
import re
from nltk import word_tokenize, pos_tag
from timeit import default_timer as timer
import numpy as np
import scipy
import ast # convert string literal list into real list.
from sklearn.metrics.pairwise import cosine_similarity
import urllib
import requests
import gensim
import warnings

"""
Below is the new endpoint dedicated for word2vec experiments.
"""
example_vec = [ -3.88111174e-02,  -6.59049079e-02,  -2.37335905e-01,
         4.83715385e-02,   9.84237939e-02,  -2.34333239e-02,
        -6.54968247e-02,  -5.68539575e-02,   1.08427159e-03,
         1.90842859e-02,  -1.13467216e-01,  -8.49631056e-03,
        -1.27356127e-01,  -5.59757203e-02,  -9.08799469e-02,
         4.41102535e-02,  -8.58965293e-02,   5.73724359e-02,
         1.56137496e-02,  -5.56014106e-02,   5.31941280e-02,
        -9.68120694e-02,   1.64142787e-01,   4.33245860e-02,
        -8.64047930e-03,  -8.68767127e-03,  -5.94879650e-02,
        -3.05731352e-02,  -7.53365457e-02,  -5.12096956e-02,
        -1.30832857e-02,  -7.89506733e-02,  -5.70434285e-03,
        -4.33039553e-02,  -1.57704558e-02,   1.53355792e-01,
        -7.89118782e-02,   7.32480586e-02,  -2.60487944e-02,
        -2.01032460e-01,  -8.78889412e-02,  -7.30841458e-02,
        -1.33337438e-01,   3.19359824e-02,   1.27773911e-01,
        -1.42432198e-01,  -2.51883902e-02,   9.62853953e-02,
         5.85342124e-02,   3.73796038e-02,   1.43504515e-02,
        -3.50148156e-02,   7.08053634e-02,   3.31287179e-03,
        -2.25382987e-02,  -1.43662840e-02,   1.58120900e-01,
        -2.36193076e-01,  -3.21339676e-03,   3.08030061e-02,
        -1.03644177e-01,   4.78657074e-02,  -9.72998738e-02,
         4.38979082e-02,   9.29532647e-02,   3.41504402e-02,
         1.23189883e-02,   2.47613788e-02,   2.48812824e-01,
        -8.68393555e-02,   3.16598117e-02,  -1.61243573e-01,
        -1.14664450e-01,  -9.06778276e-02,  -1.18543558e-01,
        -1.06279433e-01,  -4.41709114e-03,  -5.16735949e-02,
         2.58145511e-01,  -1.80289492e-01,  -2.16086060e-04,
         1.77019656e-01,  -6.12258539e-02,  -1.32735431e-01,
         8.34307075e-02,  -2.82443892e-02,  -5.25953732e-02,
        -1.04441419e-01,   9.20574833e-03,  -1.13958694e-01,
         6.13584891e-02,  -5.26212789e-02,  -1.09237731e-01,
        -1.82565197e-01,   1.47418067e-01,  -1.28216937e-01,
         9.61714983e-02,   8.90455991e-02,   2.74749361e-02,
        -1.32066935e-01,  -2.98543364e-01,  -3.06822192e-02,
        -1.27003014e-01,   2.64873859e-02,   1.17875546e-01,
        -3.29448953e-02,   1.48398697e-01,   1.77609250e-01,
         5.56902736e-02,   3.40376012e-02,  -1.71574518e-01,
         4.62812670e-02,   1.51578650e-01,  -1.03842109e-01,
        -8.53359476e-02,   1.35114893e-01,  -9.83602032e-02,
         4.66475636e-01,  -3.81718879e-03,  -2.94354036e-02,
        -5.72662093e-02,  -5.06926253e-02,   5.94390854e-02,
         7.92097226e-02,   9.89598222e-03,  -4.32951795e-03,
        -3.78126688e-02,   1.54553026e-01,   2.78610140e-01,
        -1.59583345e-01,   1.39534669e-02,  -1.63497124e-02,
        -1.28927603e-01,   8.01553503e-02,  -1.04021236e-01,
        -4.98788394e-02,  -4.01051007e-02,   2.89556414e-01,
        -2.74811387e-02,  -1.16769128e-01,  -1.41722202e-01,
        -1.39848500e-01,   6.49506673e-02,   2.59345416e-02,
         5.93397282e-02,  -5.24281934e-02,   2.20890921e-02,
         2.96231627e-01,   4.00648378e-02,  -2.39609182e-01,
         1.25380710e-01,   7.52845556e-02,  -3.13923992e-02,
        -7.06961239e-03,  -1.09664246e-01,   2.36793533e-01,
        -1.14443637e-01,  -2.89473623e-01,   1.48491599e-02,
         1.08227968e-01,   4.80440110e-02,  -2.05708388e-02,
        -1.04268387e-01,  -4.90578748e-02,   8.27955678e-02,
         1.09645016e-01,  -2.09064886e-01,   1.43066663e-02,
        -2.11853143e-02,   7.73114413e-02,   7.94803575e-02,
         1.89068280e-02,  -7.92419072e-04,  -7.02787284e-03,
        -1.07548401e-01,   1.05485758e-02,   2.51011215e-02,
         1.17356002e-01,  -1.05101146e-01,  -1.57282397e-01,
        -1.45037979e-01,   5.79039194e-02,  -5.62648326e-02,
        -1.36578023e-01,  -2.01953854e-02,   7.84947947e-02,
        -1.03510711e-02,  -9.16233808e-02,  -7.29833990e-02,
        -3.20078842e-02,   2.26762537e-02,  -1.44954294e-01,
        -7.94380233e-02,  -4.01194766e-02,   1.06817782e-01,
         2.30422720e-01,  -1.00927331e-01,  -8.57780501e-02,
         1.98180228e-01,  -2.52473384e-01]

### Google translate api credentials.
token = 'AIzaSyBZx4GANyssAEQdVlG2XuSeY-8vUsxRkBw'
regex = re.compile('[^a-zA-Z]')

### Load all vector from wordvec_file.csv
vec_dimension = len(example_vec)
null_vec = [0.0001] * vec_dimension
model = gensim.models.Word2Vec.load_word2vec_format('text8.model.bin', binary=True)
base_dir = './data/'
new_wordvec_file = base_dir + 'new_wordvec-en-vec160904.csv'
new_wordvec_file_after_retag = base_dir + 'new_retag_wordvec-en-vec160904.csv'

def get_first_nnp(pair_lst):
  nnp_counter = 0
  new_lst = []
  for word, tag in pair_lst:
      if tag == 'NNP' and nnp_counter < 2: # used to be nnp_counter 1.
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
            # query = 'http://localhost:5050/word2vec'
            # params = {'word': word}
            # res = requests.get(query, params)
            # vec = res.json()
            # if len(vec) < 10: # if return a empty list
            #     return []
            # else:                
            #     vec_single = map(lambda x: float(x.strip('\n ]')) , vec.split('[')[1].strip(']}').split(','))
            #     vec_lst.append(vec_single)
            # instead load from model, make to http request to local.
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


with open(new_wordvec_file_after_retag, 'w') as target:
    fieldnames = ['id', 'title', 'literal_title', 'tags', 'wordvec', 'url']
    writer = csv.DictWriter(target, fieldnames=fieldnames)
    writer.writeheader()

    with open(new_wordvec_file, 'r') as source:
        reader = csv.DictReader(source.read().splitlines())
        for row in reader:      
            """
            based on literal_title, update tags and wordvec fields. literal_title in Chinese.
            """
            print "writing " + row['id'] + "row..."
            literal_title = row['literal_title']

            quote = urllib.quote(literal_title)

            ### translate querystring into English.
            query = 'https://www.googleapis.com/language/translate/v2?key=' + token + '&source=zh&target=en&q=' + quote
            response = requests.get(query)
            data = response.json()
            en_querystring = data['data']['translations'][0]['translatedText'].encode('utf-8')

            ### text sanitization
            regex = re.compile('[^a-zA-Z]')
            sani_query = regex.sub(' ', en_querystring)
            sani_query = re.sub(' +',' ', sani_query).strip()

            ### nltk pos tagging
            results = filter(lambda (a,b): b in ['NN', 'NNS', 'NNP', 'NNPS'] and len(a) > 3, pos_tag(word_tokenize(sani_query)))


            results = get_first_nnp(results)
            word_vector = get_vector(results) # word_vector might be [].
            # print results, word_vector
            newrow = {'id': row['id'], 'title': row['title'], 'literal_title': row['literal_title'], 'tags': results, 'wordvec': word_vector, 'url': row['url']}
            writer.writerow(newrow)
