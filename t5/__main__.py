import argparse

from t5.experiment import Experiment


def main():
    ap = argparse.ArgumentParser("Episodic vs Step-based Framework")

    ap.add_argument("config_path", type=str,
                    help="path to config file (specifies policy representation and policy search algorithm\
                    , with relevant hyperparameters)")

    ap.add_argument("--model_path", type=str,
                    help="if given, evaluates a trained agent instead of training it")

    args = ap.parse_args()

    # configure experiment
    exp = Experiment(args.config_path)

    if args.model_path:
        exp.load(args.model_path)
        exp.test_learned()
    else:              # run experiment
        exp.run()
        exp.save()
        exp.test_learned()

   

if __name__ == "__main__":
    main()
