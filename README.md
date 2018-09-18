
## NER-LSTM-CNNs-CRF  ##
- LSTM-CNNs-CRF impolment in pytorch, and test in conll2003 dataset, reference [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf).

## Requirement ##

	pyorch: 0.3.1
	python: 3.6.1
	cuda: 8.0 (support cuda speed up, can chose)

## Usage ##
	modify the config file, detail see the Config directory
	Train:
		The best nn model will be saved during training.
		(1) sh run_train_p.sh
		(2) python -u main.py --config ./Config/config.cfg --train -p 
	Test:
		Decoding test data, and write decode result to file.
		(1) sh run_test.sh
		(2) python -u main.py --config ./Config/config.cfg --t_data test --test 
	Eval:
		For the decode result file, use conlleval script in Tools directory to calculate F-score.

## Model ##

- `BiLSTM`  
-  + `CNN`
-  + `CRF`

## Data ##

The number of sentences:  

| Data | Train | Dev | Test |  
| ------------ | ------------ | ------------ | ------------ |  
| conll2003 | 14987 | 3466 | 3684 |


- The Data format is `BIES` label, data sample in Data directory.
- Conll2003 dataset can be downloaded from [Conll2003](https://www.clips.uantwerpen.be/conll2003/ner/)
- Pre-Trained Embedding can be downloaded from [glove.6B.zip](nlp.stanford.edu/data/glove.6B.zip)

## Time ##

A simple test of the training speed and decoding time on  `GPU`.  

**GPU(GTX1080-Ti)**  

| Model | Train | Dev | Test |   
| ------------ | ------------ | ------------ | ------------ |  
| BiLSTM | 3.80s | 0.80s | 0.90s |    
| BiLSTM-CRF | 13.10s | 1.80s | 1.90s |  
| BiLSTM-CNN | 13.00s | 0.8s | 0.9s |  
| BiLSTM-CNN-CRF | 24.30s | 1.90s | 1.90s |  


## Performance ##

Performance on the `Conll2003`,  eval on the script `conlleval` in [Tools](https://github.com/bamtercelboo/pytorch_NER_PosTag_BiLSTM_CRF/tree/master/Tools)

| Model | % P | % R | % F1 |  
| ------------ | ------------ | ------------ | ------------ |  
| BLSTM | 87.78 | 87.92 | 87.85 |  
| BLSTM-CRF | 90.30 | 88.33 | 89.30 |  
| BLSTM-CNN | 88.18 | 90.30 | 89.23 |  
| BLSTM-CNN-CRF | 89.93 | 90.32 | 90.12 |  


## Reference ##
- [Ma X, and Hovy E. End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF. ACL, 2016](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf)  
- https://github.com/jiesutd/NCRFpp  
- https://github.com/liu-nlper/SLTK  


## Question ##

- if you have any question, you can open a issue or email `bamtercelboo@{gmail.com, 163.com}`.

- if you have any good suggestions, you can PR or email me.
