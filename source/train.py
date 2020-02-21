import argparse
import json
import os
import pandas as pd

import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data

from model import BinaryClassifier


def model_fn(model_dir):
    
    """Load the PyTorch model from the `model_dir` directory."""
    
    print("Loading model.")

    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BinaryClassifier(model_info['input_features'], model_info['hidden_dim'], model_info['output_dim'])

    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()
    
    print("Done loading model.")
    
    return model


def _get_train_data_loader(batch_size, training_dir):
    
    print("Get train data loader.")

    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_x = torch.from_numpy(train_data.drop([0], axis=1).values).float()

    train_ds = torch.utils.data.TensorDataset(train_x, train_y)
    data_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size)
    
    return data_loader


def train(model, train_loader, epochs, criterion, optimizer, device):
    
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    
    :param model:
        The PyTorch model that we wish to train.
    :param train_loader:
        The PyTorch DataLoader that should be used during training.
    :param epochs:
        The total number of epochs to train for.
    :param criterion: 
        The loss function used for training. 
    :param optimizer:
        The optimizer to use during training.
    :param device:
        Where the model and data should be loaded (gpu or cpu).
    """
    
    for epoch in range(1, epochs + 1):
        
        model.train()
        total_loss = 0

        for batch in train_loader:
            
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            y_pred = model(batch_x)
            
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()

        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    parser.add_argument('--batch-size', type=int, default=10, metavar='N', 
                        help='input batch size for training (default: 10)')
    
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--lr', type=int, default=.005, metavar='LR',
                        help='learning rate (default: .005)')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--input_features', type=int, default=10, metavar='IN', 
                        help='number of initial input features (default:  10)')
    
    parser.add_argument('--hidden_dim', type=int, default=24, metavar='H',
                        help='number of hidden layers in the network (default: 24)')
    
    parser.add_argument('--output_dim', type=int, default=1, metavar='O',
                        help='dimension of model output (default: 1)')
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)
    train_loader = _get_train_data_loader(args.batch_size, args.data_dir)

    model = BinaryClassifier(args.input_features, args.hidden_dim, args.output_dim).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    train(model, train_loader, args.epochs, criterion, optimizer, device)

    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim,
        }
        torch.save(model_info, f)
        
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)
