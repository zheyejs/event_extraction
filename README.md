
## NER-LSTM-CRF  ##
- LSTM-CRF impolment in pytorch, and test in conll2003 dataset.

## Requirement ##

	pyorch: 0.3.1
	python: 3.6.1
	cuda: 8.0 (support cuda speed up, can chose)

## Usage ##
	modify the config file, detail see the Config directory
	Train:
		(1)	sh run_train_p.sh
		(2)	python -u main_hyperparams.py --config ./Config/config.cfg --train -p 
	Test:
		(1) sh run_test.sh
		(2) python -u main_hyperparams.py --config ./Config/config.cfg --t_data test --test 


## Model ##

- `BiLSTM`  
- `BiLSTM-CRF`
- Now, only support `BiLSTM `, BiLSTM-CRF will be support later.

## Data ##

**The number of sentences in the two data is calculated as follows:  **  

| Data | Train | Dev | Test |  
| ------------ | ------------ | ------------ | ------------ |  
| conll2003 | 14987 | 3466 | 3684 |


- The Data format is BIES label, data sample in Data directory.
- Conll2003 dataset can be downloaded from [Conll2003](https://www.clips.uantwerpen.be/conll2003/ner/)
- Pre-Trained Embedding can be downloaded from [glove.6B.zip](nlp.stanford.edu/data/glove.6B.zip)

## Time ##

- A simple test of the training speed and decoding time on the `CPU` and `GPU`，requires only `4 seconds` for the decoding time on the GPU. why so fast ?  In terms of decoding, batch calculation is performed in some places, so the decoding time is much faster than one sentence.  
![](https://i.imgur.com/RjaG80A.jpg)

## Performance ##

- The following results are based on the neural network model of `BiLSTM + context feature`.  
![](https://i.imgur.com/mG3JyuC.jpg)

## Reference ##

- updating......

## Question ##

- if you have any question, you can open a issue or email `bamtercelboo@{gmail.com, 163.com}`.

- if you have any good suggestions, you can PR or email me.
