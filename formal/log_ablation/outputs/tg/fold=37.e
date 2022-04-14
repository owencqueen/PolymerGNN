
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
  0%|          | 0/1 [00:00<?, ?it/s]  0%|          | 0/1 [14:36<?, ?it/s]
Traceback (most recent call last):
  File "src/cv_gnn.py", line 267, in <module>
    history = CV()
  File "/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/utils/train_graphs.py", line 337, in CV_eval
    r2_test = r2_score(Ytest.numpy(), test_preds)
  File "/lustre/isaac/scratch/oqueen/polymergnn_env/lib/python3.8/site-packages/sklearn/metrics/_regression.py", line 789, in r2_score
    y_type, y_true, y_pred, multioutput = _check_reg_targets(
  File "/lustre/isaac/scratch/oqueen/polymergnn_env/lib/python3.8/site-packages/sklearn/metrics/_regression.py", line 96, in _check_reg_targets
    y_pred = check_array(y_pred, ensure_2d=False, dtype=dtype)
  File "/lustre/isaac/scratch/oqueen/polymergnn_env/lib/python3.8/site-packages/sklearn/utils/validation.py", line 800, in check_array
    _assert_all_finite(array, allow_nan=force_all_finite == "allow-nan")
  File "/lustre/isaac/scratch/oqueen/polymergnn_env/lib/python3.8/site-packages/sklearn/utils/validation.py", line 114, in _assert_all_finite
    raise ValueError(
ValueError: Input contains NaN, infinity or a value too large for dtype('float64').
