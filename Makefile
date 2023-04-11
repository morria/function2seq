init:
	pip install -r requirements.txt

clean:
	rm -rf sample/target

tensorboard:
	tensorboard dev upload --logdir sample/target/logs --name "function2seq"

split:
	python -m function2seq.split --input-file=sample/dataset.c2s --output-directory=sample/input --seed=42 --max-contexts 12 --test-ratio=0.1 --validation-ratio=0.1

vectors:
	python -m function2seq.train.vectors --input-train=sample/input/train.c2s --input-validation=sample/input/validation.c2s --output-directory=sample/target --vocab-size=1000 --text-vector-output-sequence-length=36

train:
	python -m function2seq.train --input-train=sample/input/train.c2s --input-validation=sample/input/validation.c2s --output-directory=sample/target --seed=42

predict:
	python -m function2seq.predict --target-directory=sample/target

.PHONY: init clean split train predict
