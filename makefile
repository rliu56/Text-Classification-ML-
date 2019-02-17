test: stopWords noStopWords

stopWords: NBStop LRStop PerStop

noStopWords: NBNoStop LRNoStop PerNoStop

NBStop: NB.py
	python ./NB.py ./ ./train ./test

NBNoStop: NB.py
	python ./NB.py ./NA ./train ./test

LRStop: LR.py
	python ./LR.py ./ ./train ./test -2
	python ./LR.py ./ ./train ./test -1
	python ./LR.py ./ ./train ./test 0
	python ./LR.py ./ ./train ./test 1
	python ./LR.py ./ ./train ./test 2

LRNoStop: LR.py
	python ./LR.py ./NA ./train ./test -2
	python ./LR.py ./NA ./train ./test -1
	python ./LR.py ./NA ./train ./test 0
	python ./LR.py ./NA ./train ./test 1
	python ./LR.py ./NA ./train ./test 2
  
PerStop: Perceptron.py
	python ./Perceptron.py ./ ./train ./test

PerNoStop: Perceptron.py
	python ./Perceptron.py ./NA ./train ./test