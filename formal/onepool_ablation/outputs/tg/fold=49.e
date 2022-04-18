
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


/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/utils/graph_prep.py:349: RuntimeWarning: invalid value encountered in greater
  acid_hit = (data.iloc[i,ac[0]:ac[1]].to_numpy() > 0)
/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/utils/graph_prep.py:350: RuntimeWarning: invalid value encountered in greater
  glycol_hit = (data.iloc[i,gc[0]:gc[1]].to_numpy() > 0)
  0%|          | 0/1 [00:00<?, ?it/s]  0%|          | 0/1 [00:00<?, ?it/s]
Traceback (most recent call last):
  File "src/mono_graph.py", line 259, in <module>
    history = CV()
  File "/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/utils/train_graphs.py", line 278, in CV_eval
    train_prediction = model(*make_like_batch(batch[i]), af)
  File "/lustre/isaac/scratch/oqueen/polymergnn_env/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/models/gnn/ablation_models/single_pool/tg_mono_onepool.py", line 44, in forward
    factor = self.mult_factor(x).tanh()
  File "/lustre/isaac/scratch/oqueen/polymergnn_env/lib/python3.8/site-packages/torch/nn/modules/module.py", line 1185, in __getattr__
    raise AttributeError("'{}' object has no attribute '{}'".format(
AttributeError: 'PolymerGNN_TgMono_SinglePool' object has no attribute 'mult_factor'
