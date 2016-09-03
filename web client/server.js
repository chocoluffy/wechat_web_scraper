var express = require('express');
var bodyParser = require('body-parser');
var request = require('request');
var app     = express();

/*
  Define backup file with new stamp, everytime change new stamp, others should stay unchanged.
 */
var new_stamp = "160825";

var base_dir = "./data/";
var backup = base_dir + "backup" + new_stamp + ".csv";

//Note that in version 4 of express, express.bodyParser() was
//deprecated in favor of a separate 'body-parser' module.
app.use(bodyParser.urlencoded({ extended: true })); 

//app.use(express.bodyParser());

app.get('/ada', function(req, res){
	res.sendFile('index.html', { root : __dirname});
});


app.post('/ada', function(req, res) {
	//start.js
	// var spawn = require('child_process').spawn,
	//     py    = spawn('python', ['translate.py']),
	//     
	// var data = [req.body.title, req.body.author, getDateTime(), req.body.url, req.body.description],
	    // resultString = '',
	    // log = '';
	    //
	var title = req.body.title || "Unknown",
		author = req.body.author || "Unknown",
		date = getDateTime(),
		url = req.body.url || "Unknown",
		content = req.body.description || "未知", 
		content = content + "哈", // Append Chinese characters to allow translation.
		password = req.body.password || "Unknown";

	var log = "";

	log += "The article's title: " + title + "<br>";
	log += "The author: " + author + "<br>";
	log += "The url: " + url + "<br>";
	// log += "The texts you enter: " + req.body.description + "<br>";  // for a long post, it may seems redundant.

	var logging = function(result, msg){
		result += msg;
		result += "<br>";
		return result;
	};

	/*
	  Before making request to /update, make sure stripping all newline characters from content.
	 */
	// content = content.replace(/\s+/g, '');

	/*
	  Using node's request to send GET and POST request directly to python server.
	 */
	request({ 
		url: 'http://localhost:5000/update',
		method: 'GET',
		headers: {'X-API-TOKEN' : 'FOOBAR1'},
		json: {'title': title, 'author': author, 'date': date, 'url': url, 'content': content, 'password': password}
		}, function (error, response, body) {
        if (!error && response.statusCode == 200) {
            // console.log(body);
            if (body == "nothing"){
            	log = logging(log, "<strong>By default, we won't accept short texts for now.</strong>");
            	res.send(log);
            }
            else{
	            log = logging(log, body);

	            /*
	              Sending training request.
	             */ 
	            request({ 
					url: 'http://localhost:5000/train',
					method: 'GET',
					headers: {'X-API-TOKEN' : 'FOOBAR1'},
					json: {'data-url': backup}
					}, function (error, response, body) {
					console.log(error);
			        if (!error && response.statusCode == 200) {
			            // console.log(body);
			            log = logging(log, body);

			            /*
			              Sending predicting request.
			             */
			            request({ 
							url: 'http://localhost:5000/predict',
							method: 'POST',
							headers: {'X-API-TOKEN' : 'FOOBAR1'},
							json: {'item': '-1', 'num': 3, 'data-url': backup, 'password': password}
							}, function (error, response, body) {
					        if (!error && response.statusCode == 200) {
					            // console.log(body);
					            body['info'] = log;
					            // log = logging(log, JSON.stringify(body));
					            console.log(body);
					            res.send(body);
					        }
					    });
			        }
			    });
			}
        }
    });



	/*
	  Using child process to call python scripts from local. But not working on EC2.
	 */
	// py.stdout.on('data', function(data){
	//   resultString += data.toString();
	// });
	// py.stdout.on('end', function(){
	//   // console.log('Translated texts: ', resultString);
	//   log += resultString;
	//   res.send(log);
	// });
	// py.stdin.write(JSON.stringify(data));
	// py.stdin.end();




});

app.listen(5001, function() {
  console.log('Server running at http://127.0.0.1:5001/');
});

/*
  Format current time to: "YYYY:MM:DD:HH:MM:SS".
 */
function getDateTime() {

    var date = new Date();

    var hour = date.getHours();
    hour = (hour < 10 ? "0" : "") + hour;

    var min  = date.getMinutes();
    min = (min < 10 ? "0" : "") + min;

    var sec  = date.getSeconds();
    sec = (sec < 10 ? "0" : "") + sec;

    var year = date.getFullYear();

    var month = date.getMonth() + 1;
    month = (month < 10 ? "0" : "") + month;

    var day  = date.getDate();
    day = (day < 10 ? "0" : "") + day;

    return year + ":" + month + ":" + day + ":" + hour + ":" + min + ":" + sec;

}