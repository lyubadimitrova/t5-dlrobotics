import yaml

import dict_helpers



class Experiment:

    def __init__(self, config_path):
        self.cfg = load_config(config_path)  # self.cfg is a multilevel dictionary
        self.name = self.cfg['name']

        # TODO: logging (path & tensorboard)

    def setup(self):

        self.hyperparams = self.cfg['hyperparameters']

        # TODO: custom env integration
        self.env = self.cfg['env']
        self.algo = self.cfg['algorithm']

        self.model = dict_helpers.ALGORITHMS[self.algo](self.env, **self.hyperparams)

    def run(self):
        self.model.learn(params)





def load_config(path):
    """
    Loads the configuration file on filepath into a Python dictionary.
    """
    import yaml
    
    with open(filepath, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg



env = gym.make("ALRReacherEnv")

