## Chinese-NER Task Config ##

- Use `ConfigParser` to config parameter  
	- `from configparser import ConfigParser`  
	- Detail see `config.py` and `config.cfg`, please.  

- Following is `config.cfg` Parameter details

- [Data]
	- pretrained_embed (True or False) ------ whether to use pretrained embedding

	- pretrained-embed-file (path)  ------ word embedding file path(`Pretrain_Embedding`)

	- train-file/dev-file/test-file(path)  ------ train/dev/test data path(`Data`)

	- min_freq (integer number) ------ The smallest Word frequency when build vocab

	- shuffle/epochs-shuffle(True or False) ------ shuffle data 



