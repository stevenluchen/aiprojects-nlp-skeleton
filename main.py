import os
import torch
import constants
from data.StartingDataset import StartingDataset
from networks.StartingNetwork import StartingNetwork
from train_functions.starting_train import starting_train

print('Hello World')

def main():
    # Get command line arguments
    hyperparameters = {"epochs": constants.EPOCHS, "batch_size": constants.BATCH_SIZE}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Epochs:", constants.EPOCHS)
    print("Batch size:", constants.BATCH_SIZE)

    # Initalize dataset and model. Then train the model!pytho
    data_path = "data_sets/train.csv"

    print('Prepping data now...') ######## Delete me

    train_dataset = StartingDataset(data_path)
    val_dataset = StartingDataset(data_path, is_train = False)

    print('Establishing network now...') ######## Delete me

    model = StartingNetwork()

    print('Training now...') ######## Delete me

    starting_train(
        train_dataset = train_dataset,
        val_dataset = val_dataset,
        model = model,
        hyperparameters = hyperparameters,
        n_eval = constants.N_EVAL,
    )


if __name__ == "__main__":
    main()
