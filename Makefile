init:
	pip install -r requirements.txt

test:
	py.test tests

split:
	python -m function2seq.split --input-file=sample/dataset.c2s --output-directory=sample/input --seed=42 --max-contexts 12 --test-ratio=0.1 --validation-ratio=0.1

train:
	python -m function2seq.train --input-train=sample/input/train.c2s --input-validation=sample/input/validation.c2s --output-directory=sample/model --seed=42

predict:
	python -m function2seq.prdict --model-directory=sample/model

.PHONY: init test split train predict
