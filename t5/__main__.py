import argparse

from t5.experiment import Experiment


def main():
    ap = argparse.ArgumentParser("Episodic vs Step-based Framework")

    ap.add_argument("config_path", type=str,
                    help="path to config file (specifies policy representation and policy search algorithm\
                    , with relevant hyperparameters)")

    ap.add_argument("--output_path", type=str,
                    help="path for saving learned policy and other output files", 
                    default="output/")

    args = ap.parse_args()

    
    # configure experiment
    exp = Experiment(args.config_path)

    # run experiment
    exp.run()

    # TODO: save results?
    #exp.save_final()
   

if __name__ == "__main__":
    main()
