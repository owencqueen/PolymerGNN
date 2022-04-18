
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:37<8:40:37, 637.50s/it]  4%|▍         | 2/50 [21:12<8:29:29, 636.86s/it]  6%|▌         | 3/50 [31:47<8:18:14, 636.05s/it]  8%|▊         | 4/50 [42:22<8:07:34, 635.97s/it] 10%|█         | 5/50 [52:57<7:56:45, 635.69s/it] 12%|█▏        | 6/50 [1:03:32<7:45:53, 635.30s/it] 14%|█▍        | 7/50 [1:14:06<7:35:03, 634.97s/it] 16%|█▌        | 8/50 [1:24:40<7:24:18, 634.73s/it] 18%|█▊        | 9/50 [1:35:17<7:14:09, 635.36s/it] 20%|██        | 10/50 [1:45:54<7:03:50, 635.76s/it] 22%|██▏       | 11/50 [1:56:28<6:52:58, 635.36s/it]slurmstepd: error: *** JOB 42187 ON clr1247 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
