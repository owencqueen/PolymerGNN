
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:30<8:34:40, 630.22s/it]  4%|▍         | 2/50 [20:58<8:23:45, 629.71s/it]  6%|▌         | 3/50 [31:29<8:13:32, 630.06s/it]  8%|▊         | 4/50 [42:03<8:03:51, 631.12s/it] 10%|█         | 5/50 [52:32<7:52:54, 630.55s/it] 12%|█▏        | 6/50 [1:03:03<7:42:33, 630.77s/it] 14%|█▍        | 7/50 [1:13:33<7:31:52, 630.53s/it] 16%|█▌        | 8/50 [1:24:03<7:21:10, 630.26s/it] 18%|█▊        | 9/50 [1:34:33<7:10:36, 630.17s/it] 20%|██        | 10/50 [1:45:03<7:00:09, 630.24s/it] 22%|██▏       | 11/50 [1:55:39<6:50:40, 631.80s/it]slurmstepd: error: *** JOB 42220 ON clr0823 CANCELLED AT 2022-04-14T16:18:16 DUE TO TIME LIMIT ***
