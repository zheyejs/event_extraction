
## Chinese NER  ##
- In NER task, recently, BiLSTM-CRF Neural Networks are often used, and get the best performance. but I use a simple Neural Networks(`BiLSTM`) and context feature to train data, and get a good performance close to the `BiLSTM-CRF`.

## Requirement ##

	pyorch: 0.3.1
	python: 3.6.1
	cuda: 8.0 (support cuda speed up, can chose)

## Usage ##
	modify the config file, detail see the Config directory
	(1)	sh run.sh
	(2)	python -u main_hyperparams.py --config_file ./Config/config.cfg 

## Model ##

- `BiLSTM + context feature`  
- `BiLSTM-CRF`
- Now, only support `BiLSTM + context feature`, BiLSTM-CRF will be support later.

## Data ##

## Time ##

## Performance ##

## Reference ##

## Question ##

- if you have any question, you can open a issue or email `bamtercelboo@{gmail.com, 163.com}`.

- if you have any good suggestions, you can PR or email me.
