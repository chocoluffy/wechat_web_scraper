import pandas as pd
import time
import redis
from flask import current_app, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def info(msg):
    current_app.logger.info(msg)


class ContentEngine(object):

    SIMKEY = 'p:smlr:%s'

    def __init__(self):
        self._r = redis.StrictRedis.from_url(current_app.config['REDIS_URL'])

    def train(self, data_source):
        start = time.time()
        ds = pd.read_csv(data_source)
        info("Training data ingested in %s seconds." % (time.time() - start))

        # Flush the stale training data from redis
        self._r.flushdb()

        start = time.time()
        self._train(ds)
        info("Engine trained in %s seconds." % (time.time() - start))

    def _train(self, ds):
        """
        Train the engine.

        Create a TF-IDF matrix of unigrams, bigrams, and trigrams for each product. The 'stop_words' param
        tells the TF-IDF module to ignore common english words like 'the', etc.

        Then we compute similarity between all products using SciKit Leanr's linear_kernel (which in this case is
        equivalent to cosine similarity).

        Iterate through each item's similar items and store the 100 most-similar. Stops at 100 because well...
        how many similar products do you really need to show?

        Similarities and their scores are stored in redis as a Sorted Set, with one set for each item.

        :param ds: A pandas dataset containing two fields: description & id
        :return: Nothin!
        """
        tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
        tfidf_matrix = tf.fit_transform(ds['content'])

        cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix)

        for idx, row in ds.iterrows():
            similar_indices = cosine_similarities[idx].argsort()[:-100:-1]
            similar_items = [(cosine_similarities[idx][i], ds['id'][i]) for i in similar_indices]

            # First item is the item itself, so remove it.
            # This 'sum' is turns a list of tuples into a single tuple: [(1,2), (3,4)] -> (1,2,3,4)
            flattened = sum(similar_items[1:], ())
            self._r.zadd(self.SIMKEY % row['id'], *flattened)

    def search(self, ds, results, rank):
        for indx, row in ds.iterrows():
            if str(row['id']) == results[rank][0]:
                info("NO." + str(rank) + " : " + row['title'] + " similarity score: " + str(results[rank][1]))
                break

    def predict(self, item_id, num, data_source):
        """
        Couldn't be simpler! Just retrieves the similar items and their 'score' from redis.

        :param item_id: string, if id == -1, meaning predict for the last one!!
        :param num: number of similar items to return
        :return: A list of lists like: [["19", 0.2203], ["494", 0.1693], ...]. The first item in each sub-list is
        the item ID and the second is the similarity score. Sorted by similarity score, descending.
        """
        results = self._r.zrange(self.SIMKEY % item_id, 0, num-1, withscores=True, desc=True)
        resultsLog = ''

        ds = pd.read_csv(data_source)
        for indx, row in ds.iterrows():
            if str(row['id']) == item_id:
                info("Matching for: " + row['title'])
                resultsLog += "Matching for: " + row['title']
                break

        for i in range(num):
            self.search(ds, results, i)

        predictions = []
        matches = {}
        for j in range(num):
            for index, row in ds.iterrows():
                if str(row['id']) == results[j][0]:
                    matches[str(j)] = {}
                    matches[str(j)]['title'] = str(row['title']).decode('utf-8')
                    matches[str(j)]['score'] = str(results[j][1])
                    matches[str(j)]['url'] = str(row['url']).decode('utf-8')
                    # Logging info.
                    prediction = "<br>NO." + str(j) + " : " + str(row['title']).decode('utf-8') + "<br>similarity score: " + str(results[j][1]) + "<br>url: " + str(row['url']).decode('utf-8') + "<br><br>"
                    predictions.append(prediction)

        info(predictions)
        # return "<br>".join(predictions)
        return jsonify(**matches)



content_engine = ContentEngine()