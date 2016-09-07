# -*- coding: utf-8 -*-
from flask.ext.api import FlaskAPI
from flask import request, current_app, abort, jsonify
import gensim

app = FlaskAPI(__name__)
model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)

@app.route('/word2vec')
def word2vec():
  word = request.args.get('word')
  print word
  if word in model:
    match = {"word": model[word].tolist()} 
    return jsonify(**match)
  else:
    return {"word": []}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5050)





















