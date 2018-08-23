
from configparser import ConfigParser
import os


class myconf(ConfigParser):
    def __init__(self, defaults=None):
        ConfigParser.__init__(self, defaults=defaults)
        self.add_sec = "Additional"

    def optionxform(self, optionstr):
        return optionstr


class Configurable(myconf):
    def __init__(self, config_file):
        # config = ConfigParser()
        super().__init__()

        config = myconf()
        config.read(config_file)
        # if config.has_section(self.add_sec) is False:
        #     config.add_section(self.add_sec)
        self._config = config
        self.config_file = config_file

        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, ":", v)
        if not os.path.isdir(self.save_direction):
            os.mkdir(self.save_direction)
        config.write(open(config_file, 'w'))

    def add_args(self, key, value):
        self._config.set(self.add_sec, key, value)
        self._config.write(open(self.config_file, 'w'))

    # Embed
    @property
    def pretrained_embed(self):
        return self._config.getboolean('Embed', 'pretrained_embed')

    @property
    def zeros(self):
        return self._config.getboolean('Embed', 'zeros')

    @property
    def avg(self):
        return self._config.getboolean('Embed', 'avg')

    @property
    def uniform(self):
        return self._config.getboolean('Embed', 'uniform')

    @property
    def nnembed(self):
        return self._config.getboolean('Embed', 'nnembed')

    @property
    def pretrained_embed_file(self):
        return self._config.get('Data', 'pretrained_embed_file')

    # Data
    @property
    def train_file(self):
        return self._config.get('Data', 'train_file')

    @property
    def dev_file(self):
        return self._config.get('Data', 'dev_file')

    @property
    def test_file(self):
        return self._config.get('Data', 'test_file')

    @property
    def max_count(self):
        return self._config.getint('Data', 'max_count')

    @property
    def min_freq(self):
        return self._config.getint('Data', 'min_freq')

    @property
    def shuffle(self):
        return self._config.getboolean('Data', 'shuffle')

    @property
    def epochs_shuffle(self):
        return self._config.getboolean('Data', 'epochs_shuffle')

    # Save
    @property
    def pkl_directory(self):
        return self._config.get('Save', 'pkl_directory')

    @property
    def pkl_data(self):
        return self._config.get('Save', 'pkl_data')

    @property
    def pkl_alphabet(self):
        return self._config.get('Save', 'pkl_alphabet')

    @property
    def pkl_iter(self):
        return self._config.get('Save', 'pkl_iter')

    @property
    def pkl_embed(self):
        return self._config.get('Save', 'pkl_embed')

    @property
    def save_dict(self):
        return self._config.getboolean('Save', 'save_dict')

    @property
    def save_direction(self):
        return self._config.get('Save', 'save_direction')

    @property
    def dict_directory(self):
        return self._config.get('Save', 'dict_directory')

    @property
    def word_dict(self):
        return self._config.get('Save', 'word_dict')

    @property
    def label_dict(self):
        return self._config.get('Save', 'label_dict')

    @property
    def model_name(self):
        return self._config.get('Save', 'model_name')

    @property
    def save_model(self):
        return self._config.getboolean('Save', 'save_model')

    @property
    def rm_model(self):
        return self._config.getboolean('Save', 'rm_model')

    # Model
    @property
    def model_bilstm(self):
        return self._config.getboolean("Model", "model_bilstm")

    @property
    def model_bilstm_context(self):
        return self._config.getboolean("Model", "model_bilstm_context")

    @property
    def lstm_layers(self):
        return self._config.getint("Model", "lstm_layers")

    @property
    def embed_dim(self):
        return self._config.getint("Model", "embed_dim")

    @property
    def embed_finetune(self):
        return self._config.getboolean("Model", "embed_finetune")

    @property
    def lstm_hiddens(self):
        return self._config.getint("Model", "lstm_hiddens")

    @property
    def dropout_emb(self):
        return self._config.getfloat("Model", "dropout_emb")

    @property
    def dropout(self):
        return self._config.getfloat("Model", "dropout")

    @property
    def windows_size(self):
        return self._config.getint("Model", "windows_size")

    # Optimizer
    @property
    def adam(self):
        return self._config.getboolean("Optimizer", "adam")

    @property
    def sgd(self):
        return self._config.getboolean("Optimizer", "sgd")

    @property
    def learning_rate(self):
        return self._config.getfloat("Optimizer", "learning_rate")

    @property
    def learning_rate_decay(self):
        return self._config.getfloat("Optimizer", "learning_rate_decay")

    @property
    def weight_decay(self):
        return self._config.getfloat("Optimizer", "weight_decay")

    @property
    def clip_max_norm(self):
        return self._config.getint("Optimizer", "clip_max_norm")

    # Train
    @property
    def num_threads(self):
        return self._config.getint("Train", "num_threads")

    @property
    def use_cuda(self):
        return self._config.getboolean("Train", "use_cuda")

    @property
    def set_use_cuda(self, use_cuda):
        self._config.set("Train", "use_cuda", use_cuda)

    @property
    def epochs(self):
        return self._config.getint("Train", "epochs")

    @property
    def batch_size(self):
        return self._config.getint("Train", "batch_size")

    @property
    def dev_batch_size(self):
        return self._config.getint("Train", "dev_batch_size")

    @property
    def test_batch_size(self):
        return self._config.getint("Train", "test_batch_size")

    @property
    def log_interval(self):
        return self._config.getint("Train", "log_interval")




