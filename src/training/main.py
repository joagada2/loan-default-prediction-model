import hydra
from model_evaluation import evaluate
from data_preprocessing import process_data
from model_training import train

@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(config):
    process_data(config)
    train(config)
    evaluate(config)

if __name__ == "__main__":
    main()
