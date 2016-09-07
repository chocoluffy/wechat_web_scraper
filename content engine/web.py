# -*- coding: utf-8 -*-
from flask.ext.api import FlaskAPI
from flask import request, current_app, abort, jsonify
from functools import wraps
# from textblob.blob import TextBlob
from translate import Translator
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

app = FlaskAPI(__name__)
app.config.from_object('settings')

# Everytime should only change new_stamp and stay others unchanged. 
new_stamp = '160825'


base_dir = './data/'
backup = base_dir + 'backup' + new_stamp + '.csv'
wordvec_file = base_dir + 'wordvec-en-vec160904.csv'
new_ada_content = base_dir + 'wordvec160904.csv'

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

"""
Below is the new endpoint dedicated for word2vec experiments.
"""
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

### Google translate api credentials.
token = 'AIzaSyBZx4GANyssAEQdVlG2XuSeY-8vUsxRkBw'

regex = re.compile('[^a-zA-Z]')


### Load all vector from wordvec_file.csv
allvec_lst = []
vec_dimension = len(example_vec)
null_vec = [0.0001] * vec_dimension
title_lst = []
url_lst = []
with open(wordvec_file) as source:
    reader = csv.DictReader(source.read().splitlines())
    for row in reader:      
        # notice row['wordvec'] returns a string, not a list.
        real_vec = ast.literal_eval(row['wordvec'])
        if real_vec: # can't do this, since we need the index to trace back the right entry!!
            allvec_lst.append(real_vec)
        else:
            allvec_lst.append(null_vec)

### Load Chinese titles out.
with open(new_ada_content) as source:
    reader = csv.DictReader(source.read().splitlines())
    for row in reader:
        title_lst.append(row['title'].strip().decode('utf-8'))
        url_lst.append(row['url'])

@app.route('/wordvec')
@token_auth
def wordvec():
    querystring = request.data.get('content').strip().encode('utf-8')
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
    print sani_query

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
                query = 'http://localhost:5050/word2vec'
                params = {'word': word}
                res = requests.get(query, params)
                vec = res.json()
                if len(vec) < 10: # if return a empty list
                    return []
                else:                
                    vec_single = map(lambda x: float(x.strip('\n ]')) , vec.split('[')[1].strip(']}').split(','))
                    vec_lst.append(vec_single)
                # instead load from model, make to http request to local.
                # if word in model:
                #     vec_lst.append(model[word])   
        else:
          return []
        if vec_lst:
          sum_vec = [sum(column) for column in zip(*vec_lst)]
        else:
          # in case vec_lst is empty, meaning word not in model like utevents.
          return []
        return sum_vec

    results = get_first_nnp(results)
    word_vector = get_vector(results) # word_vector might be [].
    print results, word_vector

    def cos_similarity(nested_lst, lst):
        results = map(lambda x: cosine_similarity(x, lst)[0][0], nested_lst)
        return results

    if word_vector:
        dist = cos_similarity(allvec_lst, word_vector)
        max_id = dist.index(max(dist))
        match = {}
        match['keywords'] = map(lambda x: x[0], results)
        match['title'] = title_lst[max_id]
        match['url'] = url_lst[max_id]
        return jsonify(**match)
    else:
        return {"message": "cannot find matching articles."}

if __name__ == '__main__':
    app.debug = True
    app.run()





















