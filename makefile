test: stopWords noStopWords

stopWords: NB.py
	python ./NB.py ./ ./train ./test

noStopWords: NB.py
	python ./NB.py ./not_A_Directory ./train ./test
