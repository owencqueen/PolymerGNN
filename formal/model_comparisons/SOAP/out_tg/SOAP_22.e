
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:36<8:39:59, 636.73s/it]  4%|▍         | 2/50 [21:13<8:29:22, 636.72s/it]  6%|▌         | 3/50 [31:47<8:18:08, 635.92s/it]  8%|▊         | 4/50 [42:24<8:07:41, 636.13s/it] 10%|█         | 5/50 [52:57<7:56:35, 635.46s/it] 12%|█▏        | 6/50 [1:03:34<7:46:07, 635.63s/it] 14%|█▍        | 7/50 [1:14:12<7:36:02, 636.34s/it] 16%|█▌        | 8/50 [1:24:49<7:25:35, 636.57s/it] 18%|█▊        | 9/50 [1:35:23<7:14:33, 635.93s/it] 20%|██        | 10/50 [1:45:56<7:03:23, 635.09s/it] 22%|██▏       | 11/50 [1:56:31<6:52:44, 634.97s/it]slurmstepd: error: *** JOB 42201 ON clr0710 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
