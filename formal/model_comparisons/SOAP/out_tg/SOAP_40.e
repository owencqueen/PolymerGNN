
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:37<8:40:41, 637.59s/it]  4%|▍         | 2/50 [21:14<8:29:56, 637.42s/it]  6%|▌         | 3/50 [31:51<8:19:09, 637.23s/it]  8%|▊         | 4/50 [42:26<8:08:01, 636.55s/it] 10%|█         | 5/50 [53:04<7:57:39, 636.89s/it] 12%|█▏        | 6/50 [1:03:40<7:47:02, 636.88s/it] 14%|█▍        | 7/50 [1:14:15<7:35:55, 636.18s/it] 16%|█▌        | 8/50 [1:24:49<7:24:58, 635.67s/it] 18%|█▊        | 9/50 [1:35:22<7:13:46, 634.79s/it] 20%|██        | 10/50 [1:45:55<7:02:52, 634.31s/it] 22%|██▏       | 11/50 [1:56:30<6:52:17, 634.30s/it]slurmstepd: error: *** JOB 42219 ON clr0738 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
