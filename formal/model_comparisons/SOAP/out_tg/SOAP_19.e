
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


/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/utils/graph_prep.py:226: RuntimeWarning: invalid value encountered in greater
  acid_hit = (data.iloc[i,ac[0]:ac[1]].to_numpy() > 0)
/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/utils/graph_prep.py:227: RuntimeWarning: invalid value encountered in greater
  glycol_hit = (data.iloc[i,gc[0]:gc[1]].to_numpy() > 0)
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:37<8:40:27, 637.30s/it]  4%|▍         | 2/50 [21:12<8:29:16, 636.60s/it]  6%|▌         | 3/50 [31:44<8:17:36, 635.24s/it]  8%|▊         | 4/50 [42:20<8:07:07, 635.38s/it] 10%|█         | 5/50 [52:57<7:57:03, 636.07s/it] 12%|█▏        | 6/50 [1:03:38<7:47:23, 637.34s/it] 14%|█▍        | 7/50 [1:14:11<7:35:51, 636.08s/it] 16%|█▌        | 8/50 [1:24:45<7:24:52, 635.54s/it] 18%|█▊        | 9/50 [1:35:21<7:14:22, 635.68s/it] 20%|██        | 10/50 [1:45:57<7:03:52, 635.82s/it] 22%|██▏       | 11/50 [1:56:30<6:52:44, 634.98s/it]slurmstepd: error: *** JOB 42198 ON clr0710 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
