## Chinese-NER Task Config ##

- Use `ConfigParser` to config parameter  
	- `from configparser import ConfigParser`  .
	- Detail see `config.py` and `config.cfg`, please.  

- Following is `config.cfg` Parameter details.

- [Data]
	- `pretrained_embed` (True or False) ------ whether to use pretrained embedding.

	- ` pretrained-embed-file` (path)  ------ word embedding file path(`Pretrain_Embedding`).

	- `train-file/dev-file/test-file`(path)  ------ train/dev/test data path(`Data`).

	- `min_freq` (integer number) ------ The smallest Word frequency when build vocab.

	- `shuffle/epochs-shuffle`(True or False) ------ shuffle data .

- [Save]
	- `save_direction` (path) ------ save model path.

	- `rm_model` (True or False) ------ remove model to save space(now not use).


- [Model]

	- `model-bilstm` (True or False) ------ Bilstm model.

	- `lstm-layers` (integer) ------ number layers of lstm.

	- `embed-dim` (integer) ------ embedding dim = pre-trained embedding dim.

	- `embed-finetune` (True or False) ------ word embedding finetune or no-finetune.

	- `lstm-hiddens` (integer) ------numbers of lstm hidden.

	- `dropout-emb/dropout `(float) ------ dropout for prevent overfitting.

	- `windows-size` (integer) ------ Context window feature size.
