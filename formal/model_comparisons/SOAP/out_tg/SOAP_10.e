
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:29<8:34:20, 629.80s/it]  4%|▍         | 2/50 [20:57<8:23:26, 629.29s/it]  6%|▌         | 3/50 [31:28<8:13:11, 629.61s/it]  8%|▊         | 4/50 [41:58<8:02:55, 629.90s/it] 10%|█         | 5/50 [52:28<7:52:24, 629.87s/it] 12%|█▏        | 6/50 [1:02:58<7:41:48, 629.73s/it] 14%|█▍        | 7/50 [1:13:27<7:31:14, 629.65s/it] 16%|█▌        | 8/50 [1:23:54<7:20:08, 628.77s/it] 18%|█▊        | 9/50 [1:34:21<7:09:26, 628.44s/it] 20%|██        | 10/50 [1:44:50<6:58:54, 628.37s/it] 22%|██▏       | 11/50 [1:55:19<6:48:37, 628.66s/it]slurmstepd: error: *** JOB 42189 ON clr1247 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
