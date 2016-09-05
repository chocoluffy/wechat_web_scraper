# -*- coding: utf-8 -*-
from flask.ext.api import FlaskAPI
from flask import request, current_app, abort
from functools import wraps
# from textblob.blob import TextBlob
from translate import Translator
import csv
import re
from nltk import word_tokenize, pos_tag
import gensim
from timeit import default_timer as timer
import numpy as np
import scipy
import ast # convert string literal list into real list.
from sklearn.metrics.pairwise import cosine_similarity
import urllib
import requests

app = FlaskAPI(__name__)
app.config.from_object('settings')

# Everytime should only change new_stamp and stay others unchanged. 
new_stamp = '160825'


base_dir = './data/'
backup = base_dir + 'backup' + new_stamp + '.csv'
wordvec_file = base_dir + 'wordvec-en-vec160904.csv'

def token_auth(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-TOKEN', None) != current_app.config['API_TOKEN']:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

def toUTF(text):
    return text.encode('utf-8')



@app.route('/predict', methods=['POST'])
@token_auth
def predict():
    from engines import content_engine
    item = request.data.get('item')
    realID = item
    # if item == -1, means prediction for the last row.
    if item == '-1':
        with open(backup) as source:
            reader = csv.DictReader(source.read().splitlines())
            realID = str(len(list(reader)) - 1)

    num_predictions = request.data.get('num', 10)
    data_url = request.data.get('data-url', None)
    if not realID:
        return []
    
    # For now, only returns a nested list of the top num of post and their scores, need more detailed loggin info!
    # Note that resultLst is json now. A dictionary that contains top match being jsonify.
    resultLst = content_engine.predict(str(realID), int(num_predictions), data_url)

    # If user gives uncorrect password, then just delete the last row from csv file.
    password = request.data.get('password')
    if password == 'yushunzhe':
        return resultLst
    else:   ### If password is not correct, then delete the last row from csv file. A better way is to convert csv file into list, then write back, in which case, we need to remove all space and newline from text.
        sourceLst = []
        with open(backup, 'r') as source:
            reader = csv.DictReader(source) # A list of all rows, with posts[-1] the most recent one.
            for row in reader:
                sourceLst.append(row)

        with open(backup, 'w') as target:
            fieldnames = ['id', 'title', 'author', 'date', 'url', 'content']
            writer = csv.DictWriter(target, fieldnames=fieldnames)
            writer.writeheader()
            for row in sourceLst:
                if row['id'] != str(len(list(sourceLst)) - 1):
                    newrow = {'id': row['id'], 'title': row['title'], 'author': row['author'], 'date': row['date'], 'url': row['url'], 'content': row['content']}
                    writer.writerow(newrow)
        return resultLst

@app.route('/train')
@token_auth
def train():
    from engines import content_engine
    data_url = request.data.get('data-url', None)
    content_engine.train(data_url)
    return {"message": "Success!", "success": 1}

# note that backup.csv has fields: [id,title,author,date,content].
@app.route('/update')
@token_auth
def update():
    title = request.data.get('content').encode('utf-8')
    author = request.data.get('author').encode('utf-8')
    date = request.data.get('date')
    url = request.data.get('url').encode('utf-8')
    password = request.data.get('password')
    ch_content = request.data.get('content') # in ec2 version, this part is in Chinese.

    # do some preprocess on user input here
    ch_content = "".join(ch_content.splitlines())
    # print ch_content
    # translate content into English using TextBlob, use "translate" when textblob is unavailable.
    # chinese_blob = TextBlob(ch_content)
    # translator= Translator(to_lang="en", from_lang="zh")
    # content = str(chinese_blob.translate(from_lang="zh-CN", to="en"))
    # content = str(translator.translate(ch_content.encode('utf-8'))) # "translate" module needs content input be in utf-8.
    content = translator.translate(ch_content.encode('utf-8')).encode('utf-8') # "translate" module needs content input be in utf-8.

    ### Do language sanitizing here: [1] remove stopwords. [2] stemming.
    # stop = set(stopwords.words('english'))
    # content = " ".join([i for i in content.lower().split() if i not in stop])
    # content = content
    # print content


    if content and len(content) > 10:
        with open(backup) as source:
            reader = csv.DictReader(source.read().splitlines())
            # return "number of row: " + str(len(list(reader))) # return the number of rows inside backup.csv, used as next index.
            rowid = str(len(list(reader)))
            # newrow = map(toUTF, [rowid, title, author, date, url, content])
            newrow = [rowid, title, author, date, url, content]
            with open(backup, 'a') as target:
                writer = csv.writer(target)
                writer.writerow(newrow)
                # return newrow # instead of returning that new post(look redundant), show a successful meg just be fine!
                if password == 'yushunzhe':
                    return "<strong>Your post: <" + title + "> has been succesfully uploaded to databased!!!</strong>"
                else:
                    return "<strong>Your post won't hurt database. You're in play mode.</strong>"
    else:
        return "nothing"

### Google translate api credentials.
token = 'AIzaSyBZx4GANyssAEQdVlG2XuSeY-8vUsxRkBw'

regex = re.compile('[^a-zA-Z]')

### Initializa gensim word2vec model. Around 2~3 minutes.
start = timer()
model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
end = timer()
print 'Loading word2vec model takes: ', (end - start)

@app.route('/wordvec')
@token_auth
def wordvec():
    querystring = request.data.get('content').strip().encode('utf-8')
    # print querystring
    quote = urllib.quote(querystring)

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
    print word_vector

    return {"message": "Success!", "success": 1}

if __name__ == '__main__':
    app.debug = True
    app.run()





















