#!/usr/bin/python
# -*- coding: UTF-8 -*-
import math
import operator
import sys
import glob
import re

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

# count total number of documents
def countDocs(classes,trainingDir):
	count = 0
	for c in classes:
		for filename in glob.glob(trainingDir + '/' + c + '/*.txt'):
			count += 1
	return count

# count number of documents of each class
def countDocsInClass(trainingDir, c):
	count = 0
	for filename in glob.glob(trainingDir + '/' + c + '/*.txt'):
		count += 1
	return count

# concatenate text of all documents in the same class
def concatenateTextOfAllDocsInClass(trainingDir, c, stopWords):
	textClass = []
	for filename in glob.glob(trainingDir + '/' + c + '/*.txt'):
		docFile = open(filename)
		for line in docFile:
			textClass = textClass + (re.findall(r'\w+', line)) 
		docFile.close()
	textClass = formatWords(textClass, stopWords)
	return textClass

# extract words from test document and remove those aren't in vocabulary
def extractTokensFromDoc(vocabulary, doc, stopWords):
	words = []
	docFile = open(doc)
	for line in docFile:
		words = words + (re.findall(r'\w+', line)) 
	docFile.close()
	words = list(set(formatWords(words, stopWords)))
	removeWords = []
	for word in words:
		if not (word in vocabulary): 
			removeWords.append(word)
	for word in removeWords: words.remove(word)
	return words

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

# train with training data using Multinomial Naive Bayes 
def trainMultinomialNB(classes, trainingDir, stopWords):
	prior = {}
	vocabulary = extractVocabulary(classes, trainingDir, stopWords)
	condProb = {t:{} for t in vocabulary}
	numDocs = countDocs(classes, trainingDir)	
	for c in classes:
		numClass = countDocsInClass(trainingDir, c)
		prior[c] = float(numClass) / numDocs
		textClass = concatenateTextOfAllDocsInClass(trainingDir, c, stopWords)
		tokensTerm = {}
		tokensTotal = 0
		numUniqueTerms = 0
		for t in vocabulary:
			tokensTerm[t] = textClass.count(t)
			tokensTotal += tokensTerm[t]
			if (tokensTerm[t]): numUniqueTerms += 1
		for t in vocabulary:
			condProb[t][c] = (float(tokensTerm[t]) + 1.0) / (tokensTotal + numUniqueTerms)
	return vocabulary, prior, condProb

# apply multinomial Naive Bayes to test a document
def applyMultinomialNB(classes, vocabulary, prior, condProb, d, stopWords):
	score = {}
	words = extractTokensFromDoc(vocabulary, d, stopWords)
	for c in classes:
		score[c] = math.log(prior[c])
		for t in words:
			score[c] += math.log(condProb[t][c])
	maxScoreClass = max(score.iteritems(), key = operator.itemgetter(1))[0]
	return maxScoreClass


if len(sys.argv) != 4: sys.exit()	# exit if invalid number of arguments has been input

classes = ["ham", "spam"]	# only 2 classes for this homework
stopWordsDir = sys.argv[1]	# stopWords.txt directory
trainingDir = sys.argv[2]	# training data directory
testDir = sys.argv[3]		# test data directory
stopWords = getStopWords(stopWordsDir)	# list storing stop words
vocabulary, prior, condProb = trainMultinomialNB(classes, trainingDir, stopWords)

numTestDocs = 0;	# total number of test documents
numCorrect = 0; 	# number of documents that are successfully predicted

# test every document under test folder
for c in classes:
	for doc in glob.glob(testDir + '/' + c + '/*.txt'):
		classification = applyMultinomialNB(classes, vocabulary, prior, condProb, doc, stopWords)
		numTestDocs += 1
		if (classification == c): numCorrect += 1
	
resultString = ''
if (len(stopWords) == 0): resultString = 'without '
print '\n-----------------------------------------------------------------------\n'
print '	Accuracy ' + resultString + 'throwing stop words is: ', float(numCorrect) / numTestDocs
print '\n-----------------------------------------------------------------------\n'

