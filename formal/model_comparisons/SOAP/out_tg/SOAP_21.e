
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:37<8:40:49, 637.74s/it]  4%|▍         | 2/50 [21:14<8:29:56, 637.43s/it]  6%|▌         | 3/50 [31:51<8:19:10, 637.24s/it]  8%|▊         | 4/50 [42:26<8:08:11, 636.78s/it] 10%|█         | 5/50 [53:03<7:57:30, 636.67s/it] 12%|█▏        | 6/50 [1:03:41<7:47:10, 637.06s/it] 14%|█▍        | 7/50 [1:14:18<7:36:34, 637.09s/it] 16%|█▌        | 8/50 [1:24:51<7:25:10, 635.95s/it] 18%|█▊        | 9/50 [1:35:25<7:14:09, 635.36s/it] 20%|██        | 10/50 [1:46:01<7:03:35, 635.39s/it] 22%|██▏       | 11/50 [1:56:36<6:52:58, 635.36s/it]slurmstepd: error: *** JOB 42200 ON clr0710 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
