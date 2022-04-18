
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:45<8:47:02, 645.35s/it]  4%|▍         | 2/50 [21:29<8:36:02, 645.06s/it]  6%|▌         | 3/50 [32:13<8:25:05, 644.81s/it]  8%|▊         | 4/50 [42:56<8:13:44, 644.01s/it] 10%|█         | 5/50 [53:29<8:00:32, 640.72s/it] 12%|█▏        | 6/50 [1:04:20<7:52:15, 644.00s/it] 14%|█▍        | 7/50 [1:14:52<7:38:51, 640.27s/it] 16%|█▌        | 8/50 [1:25:25<7:26:44, 638.20s/it] 18%|█▊        | 9/50 [1:36:00<7:15:20, 637.09s/it] 20%|██        | 10/50 [1:46:34<7:04:03, 636.10s/it] 22%|██▏       | 11/50 [1:57:02<6:52:04, 633.96s/it]slurmstepd: error: *** JOB 42227 ON clr1247 CANCELLED AT 2022-04-14T16:25:16 DUE TO TIME LIMIT ***
