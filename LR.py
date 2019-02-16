#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math
import operator
import sys
import glob
import re
import time
import random
from collections import Counter


# this singularization function refers to 'Inflector 2.0.11', from https://pypi.python.org/pypi/Inflector
def singularize(word) :
	'''Singularizes English nouns.'''
	
	rules = [
		['(?i)(quiz)zes$' , '\\1'],
		['(?i)(matr)ices$' , '\\1ix'],
		['(?i)(vert|ind)ices$' , '\\1ex'],
		['(?i)^(ox)en' , '\\1'],
		['(?i)(alias|status)es$' , '\\1'],
		['(?i)([octop|vir])i$' , '\\1us'],
		['(?i)(cris|ax|test)es$' , '\\1is'],
		['(?i)(shoe)s$' , '\\1'],
		['(?i)(o)es$' , '\\1'],
		['(?i)(bus)es$' , '\\1'],
		['(?i)([m|l])ice$' , '\\1ouse'],
		['(?i)(x|ch|ss|sh)es$' , '\\1'],
		['(?i)(m)ovies$' , '\\1ovie'],
		['(?i)(s)eries$' , '\\1eries'],
		['(?i)([^aeiouy]|qu)ies$' , '\\1y'],
		['(?i)([lr])ves$' , '\\1f'],
		['(?i)(tive)s$' , '\\1'],
		['(?i)(hive)s$' , '\\1'],
		['(?i)([^f])ves$' , '\\1fe'],
		['(?i)(^analy)ses$' , '\\1sis'],
		['(?i)((a)naly|(b)a|(d)iagno|(p)arenthe|(p)rogno|(s)ynop|(t)he)ses$' , '\\1\\2sis'],
		['(?i)([ti])a$' , '\\1um'],
		['(?i)(n)ews$' , '\\1ews'],
		['(?i)s$' , ''],
	];

	uncountable_words = ['equipment', 'information', 'rice', 'money', 'species', 'series', 'fish', 'sheep','sms'];

	irregular_words = {
		'people' : 'person',
		'men' : 'man',
		'children' : 'child',
		'sexes' : 'sex',
		'moves' : 'move'
	}

	for uncountable_word in uncountable_words:
		if word[-1*len(uncountable_word):] == uncountable_word :
			return word
		
	for irregular in irregular_words.keys():
		match = re.search('('+irregular+')$',word)
		if match:
			return re.sub('(?i)'+irregular+'$', match.expand('\\1')[0]+irregular_words[irregular][1:], word)
		
		for rule in range(len(rules)):
			match = re.search(rules[rule][0], word)
		if match :
			groups = match.groups()
			for k in range(0,len(groups)) :
				if groups[k] == None :
					rules[rule][1] = rules[rule][1].replace('\\'+str(k+1), '')
					
			return re.sub(rules[rule][0], rules[rule][1], word)
	
	return word
		
# delete stop words and convert words to lowercase singular form
def formatWords(words, stopWords):
	removeWords = []	# store words to be removed
	for wIndex in range(len(words)):
		words[wIndex] = words[wIndex].lower()
		if (words[wIndex] in stopWords):
			removeWords.append(words[wIndex])
	for word in removeWords: words.remove(word)
	for wIndex in range(len(words)):
			words[wIndex] = singularize(words[wIndex])
	return words

# extract vacabulary
def extractVocabulary(classes, trainingDir, stopWords):
	vocabulary = []
	for c in classes:
		for filename in glob.glob(trainingDir + '/' + c + '/*.txt'):
			docFile = open(filename)
			for line in docFile:
				vocabulary = vocabulary + (re.findall(r'\w+', line))
			docFile.close()
	return list(set(formatWords(list(set(vocabulary)), stopWords)))	# use set to remove duplicate words

# read stop words from stopWords.txt
def getStopWords(stopWordsDir):
	stopWords = []
	try:
		with open(stopWordsDir + 'stopWords.txt', 'r') as dataFile:
			stopWords = [word for line in dataFile for word in line.split()]
		dataFile.close()
	except IOError:
		pass
	return stopWords

# extract words from test document and remove those aren't in vocabulary
def extractTokensFromDoc(vocabulary, doc, stopWords):
	words = []
	docFile = open(doc)
	for line in docFile:
		words = words + (re.findall(r'\w+', line)) 
	docFile.close()
	words = formatWords(words, stopWords)
	removeWords = []
	for word in words:
		if not (word in vocabulary): 
			removeWords.append(word)
	for word in removeWords: words.remove(word)
	docFile.close()
	return words

def extractExamples(director, classes, vocabulary, stopWords):
	inputs = []
	outputs = []
	for c in classes:
		for filename in glob.glob(director + '/' + c + '/*.txt'):
			words = extractTokensFromDoc(vocabulary, filename, stopWords)
			wordCounter = Counter(words)
			inputsRow = [0.0] * (len(vocabulary) + 1)
			inputsRow[0] = 1.0
			for k,v in wordCounter.items():
				inputsRow[vocabulary.index(k) + 1] = float(v)
			inputs = inputs + [inputsRow]
			outputs = outputs + [float(classes.index(c))]
	return inputs, outputs

def sigmoid(x):
	if x < 0:
		return 1 - 1 / (1 + math.exp(x))
	return 1.0 / (1 + math.exp(-x))

def dotProduct(A, B):
	return sum([A[i] * B[i] for i in range(len(A))])

def constantProduct(n, A):
	return [n * A[i] for i in range(len(A))]
	
def matrixAddition(A, B):
	return [A[i] + B[i] for i in range(len(A))]

def matrixSubtraction(A, B):
	return [A[i] - B[i] for i in range(len(A))]
	
def trainLR(classes, trainingDir, vocabulary, stopWords, LAMBDA):
	startTime = time.time()
	weights = [0.0] * (len(vocabulary) + 1)
	inputs, outputs = extractExamples(trainingDir, classes, vocabulary, stopWords)
	
	ETA = 0.021
	for loop in range(200):
		dW = [0.0] * (len(vocabulary) + 1)
		for i in range(len(inputs)):
			estimation = sigmoid(dotProduct(inputs[i], weights))
			error= outputs[i] - estimation
			dW = matrixAddition(dW, constantProduct(error, inputs[i]))
		weights = matrixAddition(weights, matrixSubtraction(
		constantProduct(ETA, dW), constantProduct(ETA * LAMBDA, weights)))
	totalCount = len(outputs)
	print 'Training completed in %fs' % (time.time() - startTime)  
	return weights, totalCount
	
def testLR(classes, testDir, vocabulary, stopWords, weights, totalCount):
	inputs, outputs = extractExamples(testDir, classes, vocabulary, stopWords)
	correctCount = 0
	for i in range(len(inputs)):
		prediction = sigmoid(dotProduct(inputs[i], weights)) > 0.5
		if prediction == bool(outputs[i]):
			correctCount += 1
	return float(correctCount) / totalCount

if len(sys.argv) != 5: sys.exit()	# exit if invalid number of arguments has been input

classes = ["ham", "spam"]	# only 2 classes for this homework
stopWordsDir = sys.argv[1]	# stopWords.txt directory
trainingDir = sys.argv[2]	# training data directory
testDir = sys.argv[3]		# test data directory
LAMBDA = math.exp(int(sys.argv[4]))

stopWords = getStopWords(stopWordsDir)	# list storing stop words
vocabulary = extractVocabulary(classes, trainingDir, stopWords)
weights, totalCount = trainLR(classes, trainingDir, vocabulary, stopWords, LAMBDA)
accuracy = testLR(classes, testDir, vocabulary, stopWords, weights, totalCount)

resultString = 'with '
if (len(stopWords) == 0): resultString = 'without '
print '\n-----------------------------------------------------------------------\n'
print "  Lambda =", LAMBDA
print '\n	Accuracy ' + resultString + 'throwing stop words is: ', accuracy
print '\n-----------------------------------------------------------------------\n'

