import os
import torch
import numpy as np
import pandas as pd

from torch_geometric.data import Batch, Data
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def get_vector(
        data: pd.DataFrame, 
        prop: str = 'Mw (PS)', 
        fill_value: float = None,
        use_log: bool = True):
    '''
    Get vector to be added as sample-wide feature in model

    Args:
        data (pd.DataFrame): Base dataframe from which to extract the data.
        prop (str, optional): Name of column (property) for which to get the
            vector. (:default: :obj:`Mw (PS)`)
        fill_value (float, optional): Value with which to fill missing values 
            in the column. If `None`, will fill missing values with median from
            the column. (:default: :obj:`None`) 

    :rtype: pd.Series
    '''
    if fill_value is None:
        to_fill = sorted(data[prop].loc[data[prop].notna()])[int(sum(data[prop].notna())/2)]
    else:
        to_fill = fill_value
    
    if prop != '%TMP' and use_log:
        vec = np.log(data[prop].fillna(to_fill))
    else:
        # Don't log transform TMP
        vec = data[prop].fillna(to_fill)

    return vec

def get_IV_add(data):
    '''
    Return the standard IV additional data (i.e. resin properties) used
        in the paper.

    No arguments
    '''

    mw_vector = get_vector(data, prop = 'Mw (PS)').to_numpy()
    an_vector = get_vector(data, prop = 'AN').to_numpy()
    ohn_vector = get_vector(data, prop = 'OHN').to_numpy()
    tmp_vector = get_vector(data, prop = '%TMP', fill_value=0).to_numpy()

    add = np.stack([mw_vector, an_vector, ohn_vector, tmp_vector]).T

    return add

def get_Tg_add(data):
    pass

def make_like_batch(batch: tuple):
    '''
    Decomposes a batch of acid/glycol into tensors to be fed into model

    Args: 
        batch (tuple): Must be of length 2 and contain (Acid_data, Glycol_data).

    :type: tuple[`torch.geometric.data.Batch`, `torch.geometric.data.Batch`]
    '''
    Adata, Gdata = batch

    Abatch = Batch().from_data_list(Adata)
    Gbatch = Batch().from_data_list(Gdata)

    return Abatch, Gbatch 

def check_early_stop(loss_list, delay = 100):
    '''
    Checks early stopping criterion for training procedure
    Check max, see if <delay> epochs have passed since the max
    '''

    largest = np.argmin(loss_list)

    # Can enforce some smoothness condition:
    low = max(largest - 5, 0)
    up = largest + 6

    # Check if the difference between average around it and itself is different enough
    minloss = loss_list[largest]
    around_min = np.concatenate([loss_list[low:largest], loss_list[(largest+1):up]])
    smooth = np.abs(np.mean(around_min) - minloss) < np.abs(minloss * 0.25)

    return ((len(loss_list) - largest) > delay) and smooth

def train(
        model, 
        optimizer, 
        criterion, 
        dataset, 
        batch_size = 64, 
        epochs = 100
    ):

    for e in range(epochs):
        
        # Batch:
        batch, Y, add_features = dataset.get_train_batch(size = batch_size)
        test_batch, Ytest, add_test = dataset.get_test()

        train_predictions = []
        cum_loss = 0

        for i in range(batch_size):

            # Predictions:
            #predictions = torch.tensor([model(*make_like_batch(batch[i])) for i in range(batch_size)], requires_grad = True).float()
            train_prediction = model(*make_like_batch(batch[i]), torch.tensor(add_features[i]).float())
            train_predictions.append(train_prediction.clone().detach().item())
            #print(predictions)

            # Compute and backprop loss
            loss = criterion(train_prediction, torch.tensor([Y[i]]))
            optimizer.zero_grad()
            loss.backward()
            cum_loss += loss.item()
            optimizer.step()

        # Test:
        test_preds = []
        with torch.no_grad():
            for i in range(Ytest.shape[0]):
                test_preds.append(model(*make_like_batch(test_batch[i]), add_test[i]).clone().detach().item())

        r2_test = r2_score(Ytest.numpy(), test_preds)
        mse_test = mean_squared_error(Ytest.numpy(), test_preds)

        if e % 10 == 0:
            print(f'Epoch: {e}, \t Train r2: {r2_score(Y, train_predictions):.4f} \t Train Loss: {cum_loss:.4f} \t Test r2: {r2_test:.4f} \t Test Loss {mse_test:.4f}')


def CV_eval(
        dataset,
        model_generator: torch.nn.Module,
        optimizer_generator,
        criterion,
        model_generator_kwargs: dict = {},
        optimizer_kwargs: dict = {},
        batch_size = 64,
        verbose = 1,
        epochs = 1000,
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
            dataset.Kfold_CV(folds = num_folds, val = True, val_size = val_size):

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
                train_predictions.append(train_prediction.clone().detach().item())

                # Compute and backprop loss
                loss = criterion(train_prediction, torch.tensor([Y[i]]))
                optimizer.zero_grad()
                loss.backward()
                cum_loss += loss.item()
                optimizer.step()

            # Test on validation:
            model.eval()
            val_batch, Yval, add_feat_val = dataset.get_validation()
            cum_val_loss = 0
            val_preds = []
            with torch.no_grad():
                for i in range(Yval.shape[0]):
                    pred = model(*make_like_batch(val_batch[i]), add_feat_val[i])
                    val_preds.append(pred.item())
                    cum_val_loss += criterion(pred, Yval[i]).item()
                
            loss_list.append(cum_val_loss)
            model.train() # Must switch back to train after eval

            if e % 50 == 0 and (verbose == 1):
                print(f'Fold: {fold_count} \t Epoch: {e}, \
                    \t Train r2: {r2_score(Y, train_predictions):.4f} \t Train Loss: {cum_loss:.4f} \
                    Val r2: {r2_score(Yval, val_preds):.4f} \t Val Loss: {cum_val_loss:.4f}')

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
                pred = model(*make_like_batch(test_batch[i]), torch.tensor(add_test[i]).float()).clone().detach().item()
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

def train_joint(
        model, 
        optimizer, 
        criterion, 
        dataset, 
        batch_size = 64, 
        epochs = 100,
        gamma = 1e4
    ):

    for e in range(epochs):
        
        # Batch:
        batch, Y, add_features = dataset.get_train_batch(size = batch_size)
        test_batch, Ytest, add_test = dataset.get_test()

        #Y = np.log(Y)
        #Ytest = np.log(Ytest)
        # Y[:, 0] = np.log(Y[:, 0])
        # Ytest[:, 0] = np.log(Ytest[:, 0])

        train_predictions = []
        cum_loss = 0

        model.train()

        for i in range(batch_size):

            # Predictions:
            #predictions = torch.tensor([model(*make_like_batch(batch[i])) for i in range(batch_size)], requires_grad = True).float()
            train_prediction = model(*make_like_batch(batch[i]), torch.tensor(add_features[i]).float())
            #train_predictions.append(train_prediction.clone().detach().item())
            train_predictions.append([train_prediction[i].clone().detach().item() for i in ['IV', 'Tg']])
            #print(predictions)

            # Compute and backprop loss
            #loss = criterion(train_prediction, torch.tensor([Y[i]]))
            loss_IV = criterion(train_prediction['IV'], torch.tensor([Y[i][0]]))
            loss_Tg = criterion(train_prediction['Tg'], torch.tensor([Y[i][1]]))
            loss = gamma * loss_IV + loss_Tg
            optimizer.zero_grad()
            loss.backward()
            cum_loss += loss.item()
            optimizer.step()

        # Test:
        # model.eval()
        # test_predIV = []
        # test_predTg = []
        # with torch.no_grad():
        #     for i in range(Ytest.shape[0]):
        #         pred = model(*make_like_batch(test_batch[i]), torch.tensor(add_test[i]).float())
        #         test_predIV.append(pred['IV'].clone().detach().item())
        #         test_predTg.append(pred['Tg'].clone().detach().item())

        # r2_testIV = r2_score(Ytest[:,0].numpy(), test_predIV)
        # r2_testTg = r2_score(Ytest[:,1].numpy(), test_predTg)

        if e % 10 == 0:
            print(f'Epoch: {e}, \t Train r2: {r2_score(Y, train_predictions):.4f} \t Train Loss: {cum_loss:.4f}') #\t Test r2: {r2_testIV:.4f} \t Test r2 (Tg): {r2_testTg}')

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
        epochs = 1000):
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

    for test_batch, Ytest, add_test, test_inds in dataset.Kfold_CV(folds = num_folds):

        model = model_generator(**model_generator_kwargs)
        optimizer = optimizer_generator(model.parameters(), **optimizer_kwargs)

        fold_count += 1
        #Ytest = np.log(Ytest) # Log transform Ytest
        #Ytest[:, 0] = np.log(Ytest[:, 0])

        model.train()

        for e in range(epochs):
            
            # Batch:
            batch, Y, add_features = dataset.get_train_batch(size = batch_size)

            #Y = np.log(Y) # Log transform Y
            #[:, 0] = Y[:, 0])

            train_predictions = []
            cum_loss = 0

            for i in range(batch_size):

                # Predictions:
                #predictions = torch.tensor([model(*make_like_batch(batch[i])) for i in range(batch_size)], requires_grad = True).float()
                train_prediction = model(*make_like_batch(batch[i]), torch.tensor(add_features[i]).float())
                train_predictions.append([train_prediction[i].clone().detach().item() for i in ['IV', 'Tg']])
                #print(predictions)

                # Compute and backprop joint loss
                loss_IV = criterion(train_prediction['IV'], torch.tensor([Y[i][0]]))
                loss_Tg = criterion(train_prediction['Tg'], torch.tensor([Y[i][1]]))
                loss = gamma * loss_IV + loss_Tg # Loss is additive between the two
                optimizer.zero_grad()
                loss.backward()
                cum_loss += loss.item()
                optimizer.step()

            if e % 50 == 0:
                print(f'Fold: {fold_count} \t Epoch: {e}, \t Train r2: {r2_score(Y, train_predictions):.4f} \t Train Loss: {cum_loss:.4f}')


        # Test:
        model.eval()
        test_preds = []
        with torch.no_grad():
            for i in range(Ytest.shape[0]):
                #test_preds.append(model(*make_like_batch(test_batch[i]), torch.tensor(add_test[i]).float()).clone().detach().item())
                test_pred = model(*make_like_batch(test_batch[i]), torch.tensor(add_test[i]).float())
                pred = [test_pred[i].clone().detach().item() for i in ['IV', 'Tg']]
                test_preds.append(pred)
                all_predictions.append(pred)
                all_y.append(Ytest[i,:].detach().clone().tolist())
                all_reference_inds.append(test_inds[i])

        r2_test = r2_score(Ytest.numpy(), test_preds)
        r2_test_IV = r2_score(Ytest.numpy()[:, 0], np.array(test_preds)[:, 0])
        r2_test_Tg = r2_score(Ytest.numpy()[:, 1], np.array(test_preds)[:, 1])
        mse_test = mean_squared_error(Ytest.numpy(), test_preds)
        mse_test_IV = mean_squared_error(Ytest.numpy()[:, 0], np.array(test_preds)[:, 0])
        mse_test_Tg = mean_squared_error(Ytest.numpy()[:, 1], np.array(test_preds)[:, 1])
        mae_test = mean_absolute_error(Ytest.numpy(), test_preds)
        mae_test_IV = mean_absolute_error(Ytest.numpy()[:, 0], np.array(test_preds)[:, 0])
        mae_test_Tg = mean_absolute_error(Ytest.numpy()[:, 1], np.array(test_preds)[:, 1])

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

    print('Final avg. r2: ', np.mean(r2_test_per_fold))
    print('Final avg. r2 IV: ', np.mean(r2_test_per_fold_IV))
    print('Final avg. r2 Tg: ', np.mean(r2_test_per_fold_Tg))
    
    print('Final avg. MSE:', np.mean(mse_test_per_fold))
    print('Final avg. MSE IV: ', np.mean(mse_test_per_fold_IV))
    print('Final avg. MSE Tg: ', np.mean(mse_test_per_fold_Tg))

    print('Final avg. MAE:', np.mean(mae_test_per_fold))
    print('Final avg. MAE IV: ', np.mean(mae_test_per_fold_IV))
    print('Final avg. MAE Tg: ', np.mean(mae_test_per_fold_Tg))

    return all_predictions, all_y, all_reference_inds