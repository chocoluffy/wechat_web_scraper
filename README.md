Check [this post](http://docs.python-guide.org/en/latest/scenarios/scrape/) for basic python web scraping, but is is only fetching the static html page, without rending javascript part of code, since some content of the page is generated by javascript, we want to simulate a browser environment to get an complete html page for scraping.(selenium?)

From [this post](http://stackoverflow.com/questions/29449982/installing-pyqt4-with-brew), we can install `PyQt4` by doing `brew install PyQt4 --with-python27`.

The next question would be how to simulate the "load more" event? Only sliding the page to the end can trigger the load more action.

- One option is to use "selenium" to simulate the infinite scrolling event. Since we will be using chromedriver, use `brew install chromedriver` to install.

- The other one is to simple inspect the difference url being used when scrolling happens, find the pattern. But in the wechat website exmple, this trick doesn't work.

