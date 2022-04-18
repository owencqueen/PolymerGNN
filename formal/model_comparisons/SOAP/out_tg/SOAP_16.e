
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:30<8:34:38, 630.18s/it]  4%|▍         | 2/50 [20:56<8:23:17, 629.12s/it]  6%|▌         | 3/50 [31:26<8:12:57, 629.31s/it]  8%|▊         | 4/50 [41:54<8:02:05, 628.82s/it] 10%|█         | 5/50 [52:23<7:51:43, 628.97s/it] 12%|█▏        | 6/50 [1:02:52<7:41:12, 628.93s/it] 14%|█▍        | 7/50 [1:13:21<7:30:48, 629.02s/it] 16%|█▌        | 8/50 [1:23:51<7:20:26, 629.19s/it] 18%|█▊        | 9/50 [1:34:20<7:09:54, 629.14s/it] 20%|██        | 10/50 [1:44:50<6:59:35, 629.39s/it] 22%|██▏       | 11/50 [1:55:18<6:48:53, 629.06s/it]slurmstepd: error: *** JOB 42195 ON clr0823 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
