## Example Usage && Demo

- Scraping. I scrape all articles from wechat page(see below) using selenium:
![collections](http://ww3.sinaimg.cn/large/72f96cbagw1f5jos85vunj21kw0sx7bo.jpg)

- Training. `curl -X GET -H "X-API-TOKEN: <api-token>" -H "Content-Type: application/json; charset=utf-8" http://127.0.0.1:5000/train -d "{\"data-url\": \"ada-content-en.csv\"}"`. Only using a correct api-token can trigger the machine training(with the below image messege shown).
![training finished](http://ww4.sinaimg.cn/large/72f96cbagw1f5jowfjtuyj20eo00y3yl.jpg)

- Predicting. `curl -X POST -H "X-API-TOKEN: <api-token>" -H "Content-Type: application/json; charset=utf-8" http://127.0.0.1:5000/predict -d "{\"item\":1,\"num\":10,\"data-url\": \"ada-content-en.csv\"}"`. Will receive something like:
![predicting finished](http://ww1.sinaimg.cn/large/72f96cbagw1f5jozydqf6j20ym0t4aji.jpg)


## Dependency

### scraper

Install selenium, run `pip install selenium`, and chromeDriver.

### content engine

Install python library dependency in virtual environment using "conda", and redis by `brew install redis`. In order to use "flask.ext.api", we need also install Flask-API by `pip install Flask-API`.

## Logs

### scraper

Check [this post](http://docs.python-guide.org/en/latest/scenarios/scrape/) for basic python web scraping, but is is only fetching the static html page, without rending javascript part of code, since some content of the page is generated by javascript, we want to simulate a browser environment to get an complete html page for scraping.(selenium?)

From [this post](http://stackoverflow.com/questions/29449982/installing-pyqt4-with-brew), we can install `PyQt4` by doing `brew install PyQt4 --with-python27`.

The next question would be how to simulate the "load more" event? Only sliding the page to the end can trigger the load more action.

- One option is to use "selenium" to simulate the infinite scrolling event. Since we will be using chromedriver, use `brew install chromedriver` to install. One of the handy thing about selenium is that it allows us to execute javascript script on the selected elements. Check [this post](http://stackoverflow.com/questions/21006940/how-to-load-all-entries-in-an-infinite-scroll-at-once-to-parse-the-html-in-pytho) for more information of how to simulate infinite scroll.(Thus, basically with selenium we don't have to use pyqt4 webkit.)

- The other one is to simple inspect the difference url being used when scrolling happens, find the pattern. But in the wechat website exmple, this trick doesn't work.

After scraping all required static html content as string, we can do regex `findall` to find all matching urls from those three panels. Then save it to csv file. Then hand it to next step of fetching.

With all urls in place, we use beautifulsoup to extract all nested texts from article into an array, then use `filter()` to filter out elements whose parent are styles, scripts and so on. *BeautifulSoup really comes in handy for scraping all texts.* Check [this post](http://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text) for more information. One problem encountered is the encoding, I have a "ascii not recognize" error, use `.encode('utf-8')\.decode('utf-8')` to work around with that. Remember that, *when we type weird characters like Chiese characters or Abraic characters, we use utf-8,* while we better deal with or string in unicode, the procedure of from normal string to unicode is called "decode", while the way to produce string back is called "encode". Check [this post](http://stackoverflow.com/questions/5096776/unicode-decodeutf-8-ignore-raising-unicodeencodeerror) for more illustration.


### content engine

First of all, install Anaconda to create python virtualenv; go to its official website and download bash script and run it.

List all python dependencies in `conda.txt`, then run `conda create -n <virtual env's name> --file conda.txt` to create a new environment based on the library from `conda.txt`. Then the following things will be just get into that environment and get out by using: `source activate <env's name>` and `source deactivate`. Check [this post](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/) for more information about how to remove an environment, note that with conda, we can change the python version when we create conda virtual environment. Use `conda info -e` to list all current environments. 

Then, need to install redis on mac to be tested in local environment(`brew install redis`). Use `ps aux | grep redis` to check if redis server is running. If it's not running, use the following command `nohup redis-server &` to start a redis-server process and let it run in background. Check [this post](http://jasdeep.ca/2012/05/installing-redis-on-mac-os-x/) for more information about install and run redis server.

### web client

Next step, build a simple web client that allows users to upload a chunk of texts and submit, then I translate that texts into English and store in one public.csv file, then merge with ada-content-en.csv and do the TF-IDF again. Then give back a list of similar post. Web node app will spawn a child process that translate texts and construct http calls to engine server, which re-train and predict with new-coming text. Check [this post](http://www.sohamkamani.com/blog/2015/08/21/python-nodejs-comm/) for more information about how to communicate between python and node(with child_process package).

## TODOs

✔️(2016.7.4)After collecting all post url into csv file, we trace up to its pointing article page and scrape for the first three paragraphs(in case of not choking redis for too much content?), then use google translate to make it in English and do TF-IDF training. 

And for now(16.7.4), /selenium/ada.csv contains urls <s>that are repetitive</s> and <s>"wrong"(contains "&amp;" symbols e.t.c)</s>, need later update.

✔️(2016.7.6)Build a simple web client with node.js and create a sub process that runs with python script that translates it into english.

✔️(2016.7.12)Host it on AWS, airloft.org

- A fancier UI?

✔️(2016.7.16)Add a play mode. Allow user to try their own texts with our database, just not updated database. Add an auth like "author has to be ada" e.t.c.

✔️ Add field checking for forms, so that we allows users to upload empty field when in "playmode".

✔️(2016.8.21)Host a POST api request, that allows user inputs a short phase and return the most similar articles in json format.


## Side Notes

### what is conda?

Conda is a package manager application that quickly installs, runs, and updates packages and their dependencies. The conda command is the primary interface for managing installations of various packages. It can query and search the package index and current installation, create new environments, and install and update packages into existing conda environments. See our Using conda section for more information.

Conda is also an environment manager application. A conda environment is a directory that contains a specific collection of conda packages that you have installed. For example, you may have one environment with NumPy 1.7 and its dependencies, and another environment with NumPy 1.6 for legacy testing. If you change one environment, your other environments are not affected. You can easily activate or deactivate (switch between) these environments. You can also share your environment with someone by giving them a copy of your environment.yaml file. Check [this post](http://conda.pydata.org/docs/intro.html) for installation and more information.

### TextBlob as NL translation

Use TextBlob as a wrapper API for Google translation. Simple `pip install -U textblob` for installation. Try "Goslate" library, but it imposes an API query limit, which causes some inconvienience. Check [this post](https://pypi.python.org/pypi/textblob) for more information.

As an alternative, if TextBlob is down, we may switch to [translate](https://pypi.python.org/pypi/translate/) library.

Look at [translate](https://github.com/terryyin/google-translate-python) library for more information. Notice that in its issue("Need to encode input as utf-8"), we need to use like `content = str(translator.translate(ch_content.encode('utf-8')))` to get the translation working.

However, both of them are not stable enough. Use official google translate api again.

api key token: AIzaSyBZx4GANyssAEQdVlG2XuSeY-8vUsxRkBw

Then, format the query string into such format: `https://www.googleapis.com/language/translate/v2?key=<key>&source=zh&target=en&q=<words>`

### Read & Write files with .csv

Check [this post](https://docs.python.org/2/library/csv.html) for more information.

### How to improve accuracy?

Stopwords removal won't help a lot for tfidf, since those stopwords are already high-frequent words and won't be considered as keywords. However, stemming is a good choice to try.

But stopwords and stemming will help for user input! since user usually inputs a short text, which won't have duplicate words, thus each words may have higher weighted value, which causes low accuracy. However, in practive, stopwards don't help a lot for accuracy, since those stopwords occur in most of the document!

Next, several thoughts:

- word2vec with tfidf score. From tfidf algorithm, we can get each word's score, and weight words to get sentence's vector.
- gensim using LDA model. 

### How to use this api

Make a POST request to `http://airloft.org/ada/` with form key\pair values. For more details, see image below.

![example](http://ww4.sinaimg.cn/large/65e4f1e6gw1f72cadqz7tj21kw119n74.jpg)

In order to hit the endpoint from nodejs application, especially using request module, we need to pay attention to two things:
-  In request module, `body` parameter should be in string or format. Thus, if we have a json or javascript object, we can use `JSON.stringify(hash)` to transform it into string and feed it to request.
- And by default, request module sends the data in `x-www-form-urlencoded` format, which is the same as what my here "web client" server.js receives, so, such code snippet can do the tricks:

```javascript
request.post('http://airloft.org/ada/', {form:{description: inputText}}, function (err, res, body) {
        if (JSON.parse(body)["0"]) {
            // callback is the function that returns to wechat user.
            callback([
                {
                    title: JSON.parse(body)["0"]["title"],
                    url: JSON.parse(body)["0"]["url"]
                }
            ]);
        } else {
            callback('cannot find article');
        }
    });
```

### word2vec

For finding similarity, using `from sklearn.metrics.pairwise import cosine_similarity` is my final choice, it's better than simply calculating distance. 

## Trouble Shooting

### backup.csv format

After backup.csv is complete, avoid directly editing backup.csv file, any direct deleting might potentially change backup.csv format, use `git reset --hard HEAD` to rollback to previous version, do notice that such operation will discard all file changes to the previous committed version. If we want to simply discard changes on one single file, then just do `git checkout <../content\ engine/backup.csv>`.

Sometimes, if you see the result giving a 1.0 similarity match, that's probably because you may accidentally write some short texts into backup.csv file. Just edit that file and remove irrelevant info should be fine.

### start redis server

Please remember to start redis server before starting python server. For detailed information, checkout instructions above.

### Encoding error

Sometimes, we get such error as "UnicodeEncodeError: 'ascii' codec can't encode character u'\xa0' in position 20: ordinal not in range(128)", basically, stop using `str()` to convert from unicode to encoded text / bytes. Solution working in my case: replace `str()` with `.encode('utf-8')`.

### Ubuntu python module can't load

Never overuse `sudo` for `pip install`. For example, when `flask.ext.api` is deprecated, I use `Flask_API` instead. However, `sudo pip install Flask_API` doesn't solve the problem but having `pip install Flask_API` can do the trick(not `sudo pip install Flask_API`).

Also, be aware that the trick that virtual environment does is that it appends a python package path to origin $PATH env variables, so that you can access virtualenv packages like gensim from `conda install gensim`.

## Some interesting stats

What threshold confidence value will be appropriate to use to identify reliable articles?

- 期末复习资料

"score": "0.0151775943408",
"title": "UT助手数据更新：Final安排为何如此坑爹",

- 怎么准备实习
"score": "0.0158450176669",
"title": "【专访】聊聊关于PEY的那些事儿",







