import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size = batch_size, shuffle = True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):
            inputs, labels = batch
            outputs = model(inputs)

            # Backprop
            loss = loss_fn(outputs, labels)
            loss.backward()       # Compute gradients
            optimizer.step()      # Update all the weights with the gradients you just calculated
            optimizer.zero_grad() # Clear gradients before next iteration
            
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                train_accuracy, train_mean_loss = evaluate(train_loader, model, loss_fn)
                print('Train accuracy: ', train_accuracy)
                print('Train loss', train_mean_loss)

                print('Epoch:', epoch, 'Train_Loss:', loss.item())

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard. 
                # Don't forget to turn off gradient calculations!
                
                eval_accuracy, eval_mean_loss = evaluate(val_loader, model, loss_fn)
                print('Eval accuracy: ', eval_accuracy)
                print('Eval loss', eval_mean_loss)

                print('Epoch:', epoch, 'Eval_Loss: ', loss.item())
            
            step += 1

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """
    first_col, second_col = outputs.split(1, dim = 1)
    outputs = first_col < second_col
    outputs = outputs.type(torch.IntTensor).reshape(outputs.shape[0])
    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    model.eval()

    loss = 0
    n = 0

    with torch.no_grad(): # IMPORTANT: turn off gradient computations
        for batch in tqdm(val_loader):
            inputs, labels = batch
            outputs = model(inputs)
            prediction = torch.argmax(outputs)
            loss += loss_fn(outputs, labels)
            n += 1
    
    #TODO: right now the predictions are integers like 2, 32, 26, etc. shouldn't it be 0 or 1?
    # Compute accuracy
    accuracy = compute_accuracy(outputs, labels)
    mean_loss = loss / n

    model.train()

    return accuracy, mean_loss

