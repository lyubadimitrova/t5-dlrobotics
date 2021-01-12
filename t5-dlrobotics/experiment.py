import yaml

import dict_helpers



class Experiment:

    def __init__(self, config_path):
        self.cfg = load_config(config_path)  # self.cfg is a multilevel dictionary
        self.name = self.cfg['name']

    def setup(self):

        self.hyperparams = self.cfg['hyperparameters']
        self.algo = self.cfg['algorithm']

        self.model = dict_helpers.ALGORITHMS[self.algo](**self.hyperparams)


def load_config(path):
    """
    Loads the configuration file on filepath into a Python dictionary.
    """
    import yaml
    
    with open(filepath, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg
