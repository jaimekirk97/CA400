# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, jsonify, url_for

# Needed for natural language processing
import nltk
from nltk.stem.lancaster import LancasterStemmer

# Needed for tensorflow
import numpy as np
import tflearn
import tensorflow as tfl
import random

import json
import pickle
import datetime

from PIL import Image
import requests
from io import BytesIO

import enchant
import requests
from sentimentAnalysis import getSentiment, plotPieChart
import matplotlib.pyplot as plot


stemming = LancasterStemmer()

# An online tutorial was followed to help us format the our JSON data
# Link is in our repository or in appendix of technical specification

# Load response data from JSON file
with open("chatbotResponses.json") as file:
	data = json.load(file)

try:
	
	# Retrieve saved data so calculations do not need to be done again
	with open("data.pickle","rb") as f:
		words, tags, training, tag_result = pickle.load(f)

except:

	# Creating our lists
	words = []
	tags = []
	user_messages = []
	correspondingTags = []

	# We then need to gather the data in our JSON file
	# And seperate them into our lists
	for reply in data["replies"]:
		for user_message in reply["user_messages"]:
			user_message = user_message.lower()

			# Tokenize the possible user inputs into a list
			tokenized_list = nltk.word_tokenize(user_message)

			# Creates a single list with all words from all possible messages
			words.extend(tokenized_list)

			# Creates a list of lists contaning all possible user_messages
			user_messages.append(tokenized_list)

			# Appends the same tag for however many user_messages there are
			# i.e there are 7 possible user_messages with the greeting tag
			# so the word "greeting" will be appended 7 times in the list
			correspondingTags.append(reply["tag"])

		# Appends each tag only once
		if reply["tag"] not in tags:
			tags.append(reply["tag"])


	# Need to stem all our words
	words = [stemming.stem(w.lower()) for w in words if w not in "?"]

	# Remove duplicates and sort
	words = sorted(list(set(words)))

	# Sort our tags
	tags = sorted(tags)


	training = []
	tag_result = []
	# Need to convert the stemmed list of words into numerical input
	for listIndex, mess in enumerate(user_messages):
		row = [0 for _ in range(len(tags))]

		bag_of_words = []

		# Stem our user_messages into a list
		s_words = [stemming.stem(w) for w in mess]

		for w in words:
			bag_of_words.append(1) if w in s_words else bag_of_words.append(0)

		# Here we are indexing into the correspondingTags
		# list with listIndex (which is a continually increasing number)
		# And passing the result of that into the index function
		# which will return the index of that item in the tags list
		# We then make this index (which is currently 0) equal to 1
		row[tags.index(correspondingTags[listIndex])] = 1

		training.append(bag_of_words)
		tag_result.append(row)

	# Making our numpy arrays
	training = np.array(training)
	tag_result = np.array(tag_result)

	# Save our data so we do not need to do these calculations every time
	with open("data.pickle", "wb") as f:
		pickle.dump((words, tags, training, tag_result), f)


tfl.compat.v1.reset_default_graph()

# Here we are building a model for training the bot then saving it to the disk for future use
# We build multiple layers each of which have their own purpose
# The model we define here will be trained to make classifications based on user input
# e.g if the user enters "hi", then the bot will classify that as a greeting and return an appropriate response
net = tflearn.input_data(shape = [None, len(training[0])])
net = tflearn.fully_connected(net, 24)
net = tflearn.fully_connected(net, 16)
net = tflearn.fully_connected(net, 12, activation="softmax")
net = tflearn.regression(net)

# Defining the model
model = tflearn.DNN(net)


try:
	# Load a saved model
	# If we wish to retrain the model, we must first delete all model.xxx files and co. from our directory
	model.load("model.tflearn")

except:
	model = tflearn.DNN(net)
	# Begin training
	# the n_epoch variable here means the model will see the training data 1000 times
	# making it likely we will get a good accuracy on our model
	model.fit(training, tag_result, n_epoch=1000, batch_size=16, show_metric=True)
	# Save model for repeated use without the need to retrain
	model.save("model.tflearn")


# Bag of words function which is used to process our input
# An alternative to bag of words would be the word2vec model
def bagOfWords(message, words):

	# Create a numpy array of zeros the length of the words list
	bag_of_words = np.zeros(len(words), dtype=np.int_)
	
	# Tokenizing and stemming our user message
	token_words = nltk.word_tokenize(message)
	stem_words = [stemming.stem(word.lower()) for word in token_words]

	# Checking if a word in our full list of words is also present
	# in the stemmed user message
	# If so, add one to the numpy array at the current index
	for s_word in stem_words:
		for currentIndex, currentWord in enumerate(words):
			if currentWord == s_word:
				bag_of_words[currentIndex] += 1

	return bag_of_words


# Flask server
chatbotResponses = Flask(__name__)

@chatbotResponses.route("/")
def index():
	return render_template("index.html")


# Function for displaying a message when the user first loads the web page
@chatbotResponses.route("/initialResponse")
def greeting():
	return "Hello and welcome! Unsure of how to talk to me? No problem! Simply type 'help'"
	+ "and press enter to see all the ways you can interact with me! Enjoy! I look forward to speaking with you."



# Global variables to be used in our next function
info = {}
userInfo = {}
productNum = ""
productLst = []
displayResults = ""
currProd = {}
reviewInfo = {}
productReviews = {}
negativeReviews = {}
page_number = "1"
search_term = ""
total_sentiment = []


# Get message from user and match it to an appropiate set of responses
# This interacts with our javascript file
@chatbotResponses.route("/product_searching")
def get_responses():

	global productNum
	global info
	global userInfo
	global productLst
	global displayResults
	global currProd
	global reviews
	global productReviews
	global page_number
	global search_term
	global total_sentiment

	dictionary = enchant.Dict("en_US")

	userMessage = request.args.get("message")

	userMessageCheck = userMessage.split();

	commands = []

	for all_commands in data['replies']:
		for command in all_commands['user_messages']:
			commands.append(command.lower())

	if not checkInt(userMessage) == True and userMessage not in commands and 'page' not in userMessage:
		search_term = userMessage.lower()
		productLst = callToApi(search_term, page_number)
		return formatProductList(productLst)

	for word in userMessageCheck:
		if dictionary.check(word) == True and checkInt(word) != True:
			userMessage = userMessage.lower()
			# Now we make the predication for the class the user's words belong to
			# and then pick a response from that class
			# The predict funtion returns a list of probabilities for which
			# class the user message belongs to
			output = model.predict([bagOfWords(userMessage, words)])[0]

			# Returns the index of the largest probability from the list
			# This can then be used to find which tag corresponds to that
			# probability
			output_index = np.argmax(output)

			# The predicted class is found by using the output_index as
			# the index into the tags list
			tag = tags[output_index]

			# Here we are looping through all our possible replies and checking if
			# the predicted tag matches any actual tag in our response data
			for tg in data['replies']:
				try:
					options = tg['user_messages']
					if tg['tag'] == tag and tag == "greeting" or tg['tag'] == tag and tag == "name":
						replies = tg['responses']
						reply = random.choice(replies)
						break
					elif tg['tag'] == tag and tag == "helping":
						if userMessage.lower() == "help all" or userMessage.lower() == "help":
							outputHelp = "Commands: " + "^" + "To get started, simply enter a product query and wait for the list of responses to pop up" + "^" + "list Products - Generates list of products " + "^" + "price - Shows price of product currently selected " + "^" + "link - Returns link to product currently selected " + "^" + "sentiment analysis - Perform sentiment analysis on product reviews" + "^" + "ratings - return average rating and total reviews of currently selected product " + "^" + "get negative reviews - return negative reviews for given product (if any)" + "^" + "get positive reviews - return positive reviews for given product" + "^" + "next page - change product search page" + "^" + "end - end current product search" + "^"
							return outputHelp 
						elif userMessage.lower() == "help search":
							outputHelp = "Enter a product name to search existing products or type 'end' to end current search and search another product"
							return outputHelp
						elif userMessage.lower() == "help reviews":
							outputHelp = "Commands: " + "Sentiment analysis - Displays charts with review information" + "^" + "Positive - returns positive reviews of currently selected product" + "^" + "Negative - returns negative reviews of currently selected product" + "^"
							return outputHelp
					elif tg['tag'] == tag and tag == "price":
						if userMessage.lower() in options:
							if info.get(userInfo.get(int(productNum))).get("prices")[0].get("raw") == "£0.00":
								return "Product Out Of Stock"
							return "Current Price: " + info.get(userInfo.get(int(productNum))).get("prices")[0].get("raw")	
					elif tg['tag'] == tag and tag == "search":
						if userMessage.lower() in options:
							return formatProductList(productLst)
						else:
							return "Please enter the product you would like to search"	
					elif tg['tag'] == tag and tag == "rating":
						if userMessage.lower() in options:
							output = "Total ratings: " + str(info.get(userInfo.get(int(productNum))).get("ratings_total")) + " Average rating: " + str(info.get(userInfo.get(int(productNum))).get("rating"))
							return output
					elif tg['tag'] == tag and tag == "positive":
						if userMessage.lower() in options:
							asin = str(info.get(userInfo.get(int(productNum))).get("asin"))
							return getPositiveReview(asin)
					elif tg['tag'] == tag and tag == "negative":
						if userMessage.lower() in options:
							asin = str(info.get(userInfo.get(int(productNum))).get("asin"))
							return getNegativeReview(asin)
					elif tg['tag'] == tag and tag == "sentiment":
						if userMessage.lower() in options:
							asin = str(info.get(userInfo.get(int(productNum))).get("asin"))
							percentage = float(reviewAPI(asin)) * float(100)
							total_sentiment = []
							return str("{:.2f}".format(percentage)) + "% Positive"
					elif tg['tag'] == tag and tag == "link":
						if userMessage.lower() in options:
							return info.get(userInfo.get(int(productNum))).get("link")
					elif tg['tag'] == tag and tag == "page":	
						if userMessage.lower() in options:
							page_number = str(int(page_number) + 1)
							productLst = callToApi(search_term, page_number)
							return formatProductList(productLst)
					elif tg['tag'] == tag and tag == "end":
						if userMessage.lower() in options:
							page_number = "1"
							userInfo.clear()
							info.clear()
							reviewInfo.clear()
							productReviews.clear()
							productNum = None
							return "Please search another product"
						else:
							return "I'm sorry I don't understand."
				except Exception as e:
					print(e)
					return "Please enter a product number from 'List products' command"	
			return reply
		elif checkInt(word) == True:
			try:
				productNum = int(userMessage)
				currProd[userInfo.get(productNum)] = info.get(userInfo.get(productNum))
				url = info.get(userInfo.get(productNum)).get("image")
				response = requests.get(url)
				img = Image.open(BytesIO(response.content))
				img.show()
				return "What are you looking for? Enter 'help' to see a list of example commands"
			except:
				return "Please enter a product number from 'List products' command"
		else:
			return "I'm sorry, I don't understand"


def convert(str):
	converted = json.loads(str)
	return converted

def checkInt(str):
	try:
		int(str)
		return True
	except ValueError:
		return False

# Here we are formatting the returned list of products in a way that our Javascript
# file can read in and output to the webpage
def formatProductList(productLst):

	displayResults = ""
	i = 0
	while i < len(productLst) - 1:
		info[productLst[i].get("title")] = productLst[i]
		userInfo[i] = productLst[i].get("title")
		displayResults = displayResults + str(i) + ": " + userInfo[i] + "*"
		i = i + 1
	return displayResults


#Make the call to our API to get a list of products based on the user's search query
def callToApi(search_term, page_number):

	params = {

		'api_key': 'A1DD682A5E164C97999AD8CD316F2FAB',
		'type': 'search',
		'amazon_domain': 'amazon.co.uk',
		'search_term': search_term,
		'page': page_number
	}

	api_result = requests.get('https://api.rainforestapi.com/request', params)
	converted = convert(api_result.text)
	productLst = converted.get("search_results")
	return productLst


# Gather a list of reviews and perform sentiment analysis on them
# returning a pie chart with the results
def reviewAPI(asin):

	j = "1"
	total_reviews = []
	total_pages = 100
	displayResults = ""

	try:

		global total_sentiment

		while int(j) <= total_pages:

			params = {
				'api_key': 'A1DD682A5E164C97999AD8CD316F2FAB',
				'type': 'reviews',
				'amazon_domain': 'amazon.co.uk',
				'asin': asin,
				'page': j
			}

			api_result = requests.get('https://api.rainforestapi.com/request', params)
			converted = convert(api_result.text)
			reviewList = converted.get("reviews")
			total_pages = converted.get("pagination")["total_pages"]
			total_reviews = total_reviews + reviewList

			i = 0
			while i <= len(total_reviews) - 1:
				reviewInfo[total_reviews[i].get("body")] = total_reviews[i]
				productReviews[i] = total_reviews[i].get("body")
				productReviews[i] = productReviews[i].replace('"', "'")
				displayResults = productReviews[i]
				total_sentiment.append(sentiment_analysis(displayResults))
				i = i + 1

			j = str(int(j) + 1)

	except Exception as t:
		print(repr(t))


	plotPieChart(sum(total_sentiment)/len(total_sentiment))
	return str(sum(total_sentiment)/len(total_sentiment))


def getPositiveReview(asin):
	
	output = ""
	params = {
	  'api_key': 'A1DD682A5E164C97999AD8CD316F2FAB',
	  'type': 'reviews',
	  'amazon_domain': 'amazon.co.uk',
	  'asin': asin
	}

	api_result = requests.get('https://api.rainforestapi.com/request', params)
	converted = convert(api_result.text)
	reviewList = converted.get("reviews")
	reviewInfo = reviewsDict(reviewList)

	for rev in reviewInfo:
		if reviewInfo.get(rev).get("rating") > 3:
			output = output + "Title: " + reviewInfo.get(rev).get("title") + " Review: " + reviewInfo.get(rev).get("body") + " Rating: " + str(reviewInfo.get(rev).get("rating")) + "^"
	return output


def getNegativeReview(asin):

	output = ""
	params = {
	  'api_key': 'A1DD682A5E164C97999AD8CD316F2FAB',
	  'type': 'reviews',
	  'amazon_domain': 'amazon.co.uk',
	  'asin': asin
	}

	api_result = requests.get('https://api.rainforestapi.com/request', params)
	converted = convert(api_result.text)
	reviewList = converted.get("reviews")
	reviewInfo = reviewsDict(reviewList)

	for rev in reviewInfo:
		if reviewInfo.get(rev).get("rating") <= 3:
			output = output + "Title: " + reviewInfo.get(rev).get("title") + " Review: " + reviewInfo.get(rev).get("body") + " Rating: " + str(reviewInfo.get(rev).get("rating")) + "^"
	return output


def sentiment_analysis(displayResults):

	try:
		sentiment = getSentiment(displayResults)
		return sentiment

	except Exception as e:
		print(e)


def reviewsDict(reviews):
	i = 1
	for rev in reviews:
		reviewInfo[i] = rev
		i = i + 1
	return reviewInfo

if __name__ == '__main__':
	chatbotResponses.run()