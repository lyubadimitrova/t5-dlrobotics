import argparse
import os

from t5.experiment import Experiment


def main():
    ap = argparse.ArgumentParser("Episodic vs Step-based Framework")

    ap.add_argument("-c", "--config_file", type=str,
                    help="path to config file (specifies policy representation and policy search algorithm\
                    , with relevant hyperparameters)")

    ap.add_argument("-m", "--model_path", type=str,
                    help="if given, evaluates a trained agent instead of training it")

    args = ap.parse_args()

    if not args.config_file and not args.model_path:
        ap.error('Please specify a config file (-c) to train or a model path (-m) to test.')

    if args.model_path:  # overwrite is model_path is given
        args.config_file = os.path.join(args.model_path, 'config.yml')

    # configure experiment
    exp = Experiment(args.config_file)

    if args.model_path:
        exp.load(args.model_path)
        exp.test_learned()
    else:              # run experiment
        exp.run()
        exp.save()
        exp.test_learned()

   

if __name__ == "__main__":
    main()
