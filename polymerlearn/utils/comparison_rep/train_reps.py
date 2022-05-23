import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import shuffle
from polymerlearn.utils.comparison_rep import RepDataset

def CV_eval(
        dataset: RepDataset,
        model_generator: torch.nn.Module,
        optimizer_generator,
        criterion,
        model_generator_kwargs: dict = {},
        optimizer_kwargs: dict = {},
        batch_size = 64,
        verbose = 1,
        epochs = 1000,
        stop_option = 0,
        early_stop_delay = 100,
        save_state_dicts = False,
        get_scores = False,
        device = None,
        folds = 5):
    '''
    Args:
        dataset (GraphDataset): Preprocessed dataset matching the GraphDataset
            API.
        model_generator (torch.nn.Module): Class of the neural network/model that
            can be instantiated multiple times within the function.
        optimizer_generator: Optimizer that can be instantiated multiple times within
            the function.
        criterion: Loss function that can be instantiated multiple times within
            the function.
        model_generator_kwargs (dict): Dictionary of keyword arguments to be passed
            to the model for every instantiation.
        optimizer_kwargs (dict): Dictionary of keyword arguments to be passed
            to the optimizer for every instantiation.
        batch_size (int): Number of samples to be optimized on for each step. Note
            this works differently than batch size in stochastic gradient descent.
            Here, the higher value for the argument denotes more samples to be
            trained on per epoch (usually vice versa is standard).
        verbose (int): Level at which to print. Should be 0 or 1.
        epochs (int): Number of training iterations on the dataset.
        use_val (bool): If true, uses the validation set in the Dataset class.
        val_sise (float): Size of the validation set to use
        stop_option (int): Option that specifies which method to use for early
            stopping/validation saving. 0 simply performs all epochs for each fold.
            1 performs all epochs but uses model with highest validation score for 
            evaluation on test set. 2 stops early if the validation loss was at least
            `early_stop_delay` epochs ago; it loads that trial's model and evaluates
            on it.
        early_stop_delay (int): Number of epochs to wait after an early stopping condition
            is met.
        save_state_dicts (bool): If True, returns state dictionaries for the model at
            each fold. Useful for explainability.
        get_scores (bool, optional): If True, return only the average values of metrics 
            across the folds
        device (str): Device name at which to run torch calculations on. Supports GPU.
    '''

    num_folds = folds
    fold_count = 0

    r2_test_per_fold = []
    mse_test_per_fold = []
    mae_test_per_fold = []

    all_predictions = []
    all_y = []
    all_reference_inds = []

    model_state_dicts = []

    for Xtrain, Xtest, ytrain, ytest, train_idx, test_idx in dataset.Kfold_CV(folds = num_folds):

        # Instantiate fold-level model and optimizer:
        model = model_generator(**model_generator_kwargs).to(device)
        # Move model to GPU before setting optimizer
        optimizer = optimizer_generator(model.parameters(), **optimizer_kwargs)

        fold_count += 1
        loss_list = []

        for e in range(epochs):

            train_predictions = []
            cum_loss = 0

            Xtrain, ytrain = shuffle(Xtrain, ytrain)

            for i in range(batch_size):

                if verbose > 1:
                    print('Additional it={}'.format(i), af)
                train_prediction = model(Xtrain[i,:])
                if verbose > 1:
                    print('pred', train_prediction.item())
                train_predictions.append(train_prediction.clone().detach().item())

                # Compute and backprop loss
                loss = criterion(train_prediction, ytrain[i])
                optimizer.zero_grad()
                loss.backward()
                cum_loss += loss.item()
                optimizer.step()

            if verbose > 1:
                print('Train predictions', train_predictions)

            if e % 50 == 0 and (verbose >= 1):
                print_str = f'Fold: {fold_count} \t Epoch: {e}, \
                    \t Train r2: {r2_score(Y, train_predictions):.4f} \t Train Loss: {cum_loss:.4f}' 
                if use_val:
                   print_str += f'Val r2: {r2_score(Yval, val_preds):.4f} \t Val Loss: {cum_val_loss:.4f}'
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
            for i in range(ytest.size[0]):
                at = None if add_test is None else torch.tensor(add_test[i]).float()
                pred = model(Xtest[i]).clone().detach().item()
                test_preds.append(pred)
                all_predictions.append(pred)
                all_y.append(ytest[i].item())
                all_reference_inds.append(test_idx[i])

        r2_test = r2_score(ytest.cpu().numpy(), test_preds)
        mse_test = mean_squared_error(ytest.cpu().numpy(), test_preds)
        mae_test = mean_absolute_error(ytest.cpu().numpy(), test_preds)

        print(f'Fold: {fold_count} \t Test r2: {r2_test:.4f} \t Test Loss: {mse_test:.4f} \t Test MAE: {mae_test:.4f}')

        r2_test_per_fold.append(r2_test)
        mse_test_per_fold.append(mse_test)
        mae_test_per_fold.append(mae_test)

        if save_state_dicts:
            model_state_dicts.append(model.state_dict())


    print('Final avg. r2: ', np.mean(r2_test_per_fold))
    print('Final avg. MSE:', np.mean(mse_test_per_fold))
    print('Final avg. MAE:', np.mean(mae_test_per_fold))

    r2_avg = np.mean(r2_test_per_fold)
    mae_avg = np.mean(mae_test_per_fold)

    big_ret_dict = {
        'r2': r2_avg,
        'mae': mae_avg,
        'all_predictions': all_predictions,
        'all_y': all_y,
        'all_reference_inds': all_reference_inds,
        'model_state_dicts': model_state_dicts
    }

    if save_state_dicts:
        if get_scores:
            return big_ret_dict
        else:
            return all_predictions, all_y, all_reference_inds, model_state_dicts

    if get_scores: # Return scores:
        return big_ret_dict

    return all_predictions, all_y, all_reference_inds


def CV_eval_joint(
        dataset,
        model_generator: torch.nn.Module,
        optimizer_generator,
        criterion,
        model_generator_kwargs: dict = {},
        optimizer_kwargs: dict = {},
        batch_size = 64,
        verbose = 1,
        gamma = 1e4,
        epochs = 1000,
        get_scores = False,
        device = None,
        save_state_dicts = False):
    '''
    Cross validation of the joint Tg/IV model

    Args:
        gamma (float): Weighting factor applied to IV loss. Used
            to balance the losses between IV and Tg during the joint
            training process.
    '''

    num_folds = 5
    fold_count = 0

    r2_test_per_fold = []
    r2_test_per_fold_IV = []
    r2_test_per_fold_Tg = []

    mse_test_per_fold = []
    mse_test_per_fold_IV = []
    mse_test_per_fold_Tg = []

    mae_test_per_fold = []
    mae_test_per_fold_IV = []
    mae_test_per_fold_Tg = []

    all_predictions = []
    all_y = []
    all_reference_inds = []

    model_state_dicts = []

    for Xtrain, Xtest, ytrain, ytest, train_idx, test_idx in dataset.Kfold_CV(folds = num_folds):

        model = model_generator(**model_generator_kwargs).to(device)
        optimizer = optimizer_generator(model.parameters(), **optimizer_kwargs)

        fold_count += 1
        model.train()

        e = 0
        while True:

            train_predictions = []
            cum_loss = 0

            # Shuffle Xtrain, ytrain:
            Xtrain, ytrain = shuffle(Xtrain, ytrain)

            for i in range(batch_size):
                train_prediction = model(Xtrain[i])
                train_predictions.append([train_prediction[i].clone().detach().item() for i in ['IV', 'Tg']])

                # Compute and backprop joint loss
                loss_IV = criterion(train_prediction['IV'], torch.tensor([ytrain[i][0]]).to(device))
                loss_Tg = criterion(train_prediction['Tg'], torch.tensor([ytrain[i][1]]).to(device))
                loss = gamma * loss_IV + loss_Tg # Loss is additive between the two
                optimizer.zero_grad()
                loss.backward()
                cum_loss += loss.item()
                optimizer.step()
                
            try:
                r2IV = r2_score(ytrain[:][0], train_predictions[0][:])
            except:
                r2IV = -1
            try:
                r2Tg = r2_score(ytrain[:][1], train_predictions[1][:])
            except:
                r2Tg = -1

            if e % 50 == 0:
                #print(f'Fold: {fold_count} \t Epoch: {e}, \t Train r2: {r2_score(Y, train_predictions):.4f} \t Train Loss: {cum_loss:.4f}')
                print(f'Fold: {fold_count} : {e}, Train r2 IV, Tg: {r2IV:.4f}, {r2Tg:.4f} \t Train Loss: {cum_loss:.4f}')
                
            if e > epochs and (r2IV > 0.9) and (r2Tg > 0.9):
                # Check for stable learning on both IV and Tg
                # Checks traning value, not validation
                break
            
            e += 1

        # Test after fold trains:
        model.eval()
        test_preds = []
        with torch.no_grad():
            for i in range(ytest.shape[0]):
                test_pred = model(Xtest[i])
                pred = [test_pred[i].clone().detach().item() for i in ['IV', 'Tg']]
                test_preds.append(pred)
                all_predictions.append(pred)
                all_y.append(ytest[i,:].detach().clone().tolist())
                all_reference_inds.append(test_idx[i])

        r2_test = r2_score(ytest.cpu().numpy(), test_preds)
        r2_test_IV = r2_score(ytest.cpu().numpy()[:, 0], np.array(test_preds)[:, 0])
        r2_test_Tg = r2_score(ytest.cpu().numpy()[:, 1], np.array(test_preds)[:, 1])
        mse_test = mean_squared_error(ytest.cpu().numpy(), test_preds)
        mse_test_IV = mean_squared_error(ytest.cpu().numpy()[:, 0], np.array(test_preds)[:, 0])
        mse_test_Tg = mean_squared_error(ytest.cpu().numpy()[:, 1], np.array(test_preds)[:, 1])
        mae_test = mean_absolute_error(ytest.cpu().numpy(), test_preds)
        mae_test_IV = mean_absolute_error(ytest.cpu().numpy()[:, 0], np.array(test_preds)[:, 0])
        mae_test_Tg = mean_absolute_error(ytest.cpu().numpy()[:, 1], np.array(test_preds)[:, 1])

        print(f'Fold: {fold_count} \t Test r2: {r2_test:.4f} \t r2_IV: {r2_test_IV:.4f} \t r2_Tg: {r2_test_Tg:.4f} \t MSE: {mse_test:.4f} \t MSE_IV: {mse_test_IV:.4f} \t MSE_Tg: {mse_test_Tg:.4f} \t MAE: {mae_test:.4f} \t MAE_IV: {mae_test_IV:.4f} \t MAE_Tg: {mae_test_Tg:.4f}')

        r2_test_per_fold.append(r2_test)
        r2_test_per_fold_IV.append(r2_test_IV)
        r2_test_per_fold_Tg.append(r2_test_Tg)

        mse_test_per_fold.append(mse_test)
        mse_test_per_fold_IV.append(mse_test_IV)
        mse_test_per_fold_Tg.append(mse_test_Tg)

        mae_test_per_fold.append(mae_test)
        mae_test_per_fold_IV.append(mae_test_IV)
        mae_test_per_fold_Tg.append(mae_test_Tg)

        if save_state_dicts:
            model_state_dicts.append(model.state_dict())

    print('Final avg. r2: ', np.mean(r2_test_per_fold))
    print('Final avg. r2 IV: ', np.mean(r2_test_per_fold_IV))
    print('Final avg. r2 Tg: ', np.mean(r2_test_per_fold_Tg))
    
    print('Final avg. MSE:', np.mean(mse_test_per_fold))
    print('Final avg. MSE IV: ', np.mean(mse_test_per_fold_IV))
    print('Final avg. MSE Tg: ', np.mean(mse_test_per_fold_Tg))

    print('Final avg. MAE:', np.mean(mae_test_per_fold))
    print('Final avg. MAE IV: ', np.mean(mae_test_per_fold_IV))
    print('Final avg. MAE Tg: ', np.mean(mae_test_per_fold_Tg))

    d = {
        'IV':(np.mean(r2_test_per_fold_IV), np.mean(mae_test_per_fold_IV)),
        'Tg':(np.mean(r2_test_per_fold_Tg), np.mean(mae_test_per_fold_Tg)),
        'all_predictions': all_predictions,
        'all_y': all_y,
        'all_reference_inds': all_reference_inds,
        'model_state_dicts': model_state_dicts
    }

    if save_state_dicts:
        if get_scores:
            return d
        else:
            return all_predictions, all_y, all_reference_inds, model_state_dicts

    if get_scores:
        # Return in a dictionary
        return d

    return all_predictions, all_y, all_reference_inds