
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
To initialize your shell, run

    $ conda init <SHELL_NAME>

Currently supported shells are:
  - bash
  - fish
  - tcsh
  - xonsh
  - zsh
  - powershell

See 'conda init --help' for more information and options.

IMPORTANT: You may need to close and restart your shell after running 'conda init'.


/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/utils/graph_prep.py:337: RuntimeWarning: invalid value encountered in greater
  acid_hit = (data.iloc[i,ac[0]:ac[1]].to_numpy() > 0)
/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/utils/graph_prep.py:338: RuntimeWarning: invalid value encountered in greater
  glycol_hit = (data.iloc[i,gc[0]:gc[1]].to_numpy() > 0)
  0%|          | 0/1 [00:00<?, ?it/s]/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/utils/train_graphs.py:500: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at  /opt/conda/conda-bld/pytorch_1646755853042/work/torch/csrc/utils/tensor_new.cpp:210.)
  add_features = torch.tensor(add_features).float()
  0%|          | 0/1 [03:37<?, ?it/s]
Traceback (most recent call last):
  File "src/cv_gnn.py", line 239, in <module>
    history = CV()
  File "/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/utils/train_graphs.py", line 527, in CV_eval_joint
    print(f'Fold: {fold_count} \t Epoch: {e}, \t Train r2: {r2_score(Y, train_predictions):.4f} \t Train Loss: {cum_loss:.4f}')
  File "/lustre/isaac/scratch/oqueen/polymergnn_env/lib/python3.8/site-packages/sklearn/metrics/_regression.py", line 789, in r2_score
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
  File "/lustre/isaac/scratch/oqueen/polymergnn_env/lib/python3.8/site-packages/sklearn/metrics/_regression.py", line 96, in _check_reg_targets
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
  File "/lustre/isaac/scratch/oqueen/polymergnn_env/lib/python3.8/site-packages/sklearn/utils/validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "/lustre/isaac/scratch/oqueen/polymergnn_env/lib/python3.8/site-packages/sklearn/utils/validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
