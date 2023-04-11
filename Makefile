init:
	pip install -r requirements.txt

test:
	py.test tests

tensorboard:
	tensorboard dev upload --logdir sample/model/logs --name "function2seq"

split:
	python -m function2seq.split --input-file=sample/dataset.c2s --output-directory=sample/input --seed=42 --max-contexts 12 --test-ratio=0.1 --validation-ratio=0.1

vectors:
	python -m function2seq.train.vectors --input-train=sample/input/train.c2s --input-validation=sample/input/validation.c2s --output-directory=sample/model --vocab-size=1000 --text-vector-output-sequence-length=36

train:
	python -m function2seq.train --input-train=sample/input/train.c2s --input-validation=sample/input/validation.c2s --output-directory=sample/model --seed=42

predict:
	python -m function2seq.predict --model-directory=sample/model

.PHONY: init test split train predict
