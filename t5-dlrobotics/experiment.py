import yaml

from constants import *



class Experiment:

    def __init__(self, config_path):
        self.cfg = load_config(config_path)  # self.cfg is a multilevel dictionary
        self.name = self.cfg['name']


def load_config(path):
    """
    Loads the configuration file on filepath into a Python dictionary.
    """
    import yaml
    
    with open(filepath, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg
