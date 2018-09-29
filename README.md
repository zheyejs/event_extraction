
## NER-LSTM-CNNs-CRF  ##
- LSTM-CNNs-CRF impolment in pytorch, and test in conll2003 dataset, reference [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](http://www.aclweb.org/anthology/P/P16/P16-1101.pdf).

## Requirement ##

	pyorch: 0.3.1(pip install torch==0.3.1)
	python: 3.6.1
	cuda: 8.0 or 9.0(support cuda speed up, can chose)

[**pytorch 0.4.0**  refer to [pytorch0.4.0error #2](https://github.com/bamtercelboo/pytorch_NER_BiLSTM_CNN_CRF/issues/2)]

## Usage ##
	modify the config file, detail see the Config directory
	Train:
		The best nn model will be saved during training.
		(1) sh run_train_p.sh
		(2) python main.py --config ./Config/config.cfg --train -p 
	Test:
		Decoding test data, and write decode result to file.
		(1) sh run_test.sh
		(2) python main.py --config ./Config/config.cfg --t_data test --test 
	Eval:
		For the decode result file, use conlleval script in Tools directory to calculate F-score.

## Config ##
	[Embed]
		pretrained_embed = True（default: False）
		nnembed = True
		pretrained_embed_file = embed file path
	[Data]
		max_count = -1  ## Number of sentences loaded(-1 represents all)
	[Save]
		save_pkl = True(save pkl file for test, default True)
	[Model]
		use_crf = False  ## CRF 
		use_char = True  ## CNN
		model_bilstm = True  ##BiLSTM
		embed_dim = 100, lstm_hiddens = 100
		dropout_emb = 0.5, dropout = 0.5
		max_char_len = 20, char_dim = 30, conv_filter_sizes = 3, conv_filter_nums = 30
	[Optimizer]
		sgd = True
		learning_rate = 0.015 , weight_decay = 1.0e-8
		use_lr_decay = True, lr_rate_decay = 0.05, min_lrate = 0.000005, max_patience = 1
	[Train]
		use_cuda = True (choose cuda speed up, default is True)
		early_max_patience = 10(early stop max patience)

This is a major configuration file description, for more detailed reference to config.cfg file and [config readme](https://github.com/bamtercelboo/pytorch_NER_BiLSTM_CNN_CRF/tree/master/Config).

## Model ##

- `BiLSTM`  
	- `CNN`
	-  `CRF`

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
