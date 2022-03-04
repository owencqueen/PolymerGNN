import os
import torch
import numpy as np
import pandas as pd

from torch_geometric.data import Batch, Data
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from polymerlearn.utils import make_like_batch
from polymerlearn.utils.train_graphs import check_early_stop\

from polymerlearn.utils.losses import evidential_loss


def clone_dict(d):
    '''
    Clones all elements (assumed torch.Tensors) of dictionary d
    '''

    clone_dict = {}
    for k, v in d:
        clone_dict[k] = v.detach().clone().item()

    return clone_dict

def CV_eval_evidential(
        dataset,
        model_generator: torch.nn.Module,
        optimizer_generator,
        model_generator_kwargs: dict = {},
        optimizer_kwargs: dict = {},
        batch_size = 64,
        verbose = 1,
        epochs = 1000,
        use_val = False,
        val_size = 0.1,
        stop_option = 0,
        early_stop_delay = 100):
    '''
    
    Args:
        stop_option (int): Option that specifies which method to use for early
            stopping/validation saving. 0 simply performs all epochs for each fold.
            1 performs all epochs but uses model with highest validation score for 
            evaluation on test set. 2 stops early if the validation loss was at least
            `early_stop_delay` epochs ago; it loads that trial's model and evaluates
            on it.
    
    '''

    num_folds = 5
    fold_count = 0

    r2_test_per_fold = []
    mse_test_per_fold = []
    mae_test_per_fold = []

    all_predictions = []
    all_y = []
    all_reference_inds = []

    for test_batch, Ytest, add_test, test_inds in \
            dataset.Kfold_CV(folds = num_folds, val = use_val, val_size = val_size):

        # Instantiate fold-level model and optimizer:
        model = model_generator(**model_generator_kwargs)
        optimizer = optimizer_generator(model.parameters(), **optimizer_kwargs)

        fold_count += 1
        loss_list = []

        if stop_option >= 1:
            min_val_loss = 1e10
            min_val_state_dict = None

        for e in range(epochs):
            
            # Bootstrap batches:
            batch, Y, add_features = dataset.get_train_batch(size = batch_size)

            train_predictions = []
            cum_loss = 0

            for i in range(batch_size):

                # Predictions:
                train_prediction = model(*make_like_batch(batch[i]), torch.tensor(add_features[i]).float())
                train_predictions.append(train_prediction['gamma'].detach().clone().item())

                # Compute and backprop loss
                #loss = criterion(train_prediction, torch.tensor([Y[i]]))
                loss = evidential_loss(torch.tensor([Y[i]]), 
                    output_dict=train_prediction,
                    coef = 1)
                optimizer.zero_grad()
                loss.backward()
                cum_loss += loss.item()
                optimizer.step()

            # Test on validation:
            if use_val:
                model.eval()
                val_batch, Yval, add_feat_val = dataset.get_validation()
                cum_val_loss = 0
                val_preds = []
                with torch.no_grad():
                    for i in range(Yval.shape[0]):
                        pred = model(*make_like_batch(val_batch[i]), add_feat_val[i])
                        val_preds.append(pred['gamma'].detach().clone().item())
                        cum_val_loss += evidential_loss(Yval[i], pred, coef = 1).item()
                    
                loss_list.append(cum_val_loss)
                model.train() # Must switch back to train after eval

            if e % 50 == 0 and (verbose == 1):
                print_str = f'Fold: {fold_count} \t Epoch: {e}, \
                    \t Train r2: {r2_score(Y, train_predictions):.4f} \t Train Loss: {cum_loss:.4f}'
                if use_val:
                    print_str += f'\t Val r2: {r2_score(Yval, val_preds):.4f} \t Val Loss: {cum_val_loss:.4f}'
                print(print_str)

            if stop_option >= 1:
                if cum_val_loss < min_val_loss:
                    # If min val loss, store state dict
                    min_val_loss = cum_val_loss
                    min_val_state_dict = model.state_dict()

            # Check early stop if needed:
            if stop_option == 2:
                # Check criteria:
                if check_early_stop(loss_list, early_stop_delay) and e > early_stop_delay:
                    break

        
        if stop_option >= 1: # Loads the min val loss state dict even if we didn't break
            # Load in the model with min val loss
            model = model_generator(**model_generator_kwargs)
            model.load_state_dict(min_val_state_dict)

        # Test:
        test_preds = []
        with torch.no_grad():
            for i in range(Ytest.shape[0]):
                pred = model(*make_like_batch(test_batch[i]), torch.tensor(add_test[i]).float())['gamma'].clone().detach().item()
                test_preds.append(pred)
                all_predictions.append(pred)
                all_y.append(Ytest[i].item())
                all_reference_inds.append(test_inds[i])

        r2_test = r2_score(Ytest.numpy(), test_preds)
        mse_test = mean_squared_error(Ytest.numpy(), test_preds)
        mae_test = mean_absolute_error(Ytest.numpy(), test_preds)

        print(f'Fold: {fold_count} \t Test r2: {r2_test:.4f} \t Test Loss: {mse_test:.4f} \t Test MAE: {mae_test:.4f}')

        r2_test_per_fold.append(r2_test)
        mse_test_per_fold.append(mse_test)
        mae_test_per_fold.append(mae_test)

    print('Final avg. r2: ', np.mean(r2_test_per_fold))
    print('Final avg. MSE:', np.mean(mse_test_per_fold))
    print('Final avg. MAE:', np.mean(mae_test_per_fold))

    return all_predictions, all_y, all_reference_inds