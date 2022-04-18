
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:34<8:38:27, 634.85s/it]  4%|▍         | 2/50 [21:06<8:27:04, 633.84s/it]  6%|▌         | 3/50 [31:51<8:19:11, 637.26s/it]  8%|▊         | 4/50 [42:23<8:07:22, 635.71s/it] 10%|█         | 5/50 [53:01<7:57:21, 636.47s/it] 12%|█▏        | 6/50 [1:03:33<7:45:38, 634.97s/it] 14%|█▍        | 7/50 [1:14:08<7:35:04, 634.99s/it] 16%|█▌        | 8/50 [1:24:45<7:25:00, 635.72s/it] 18%|█▊        | 9/50 [1:35:19<7:13:57, 635.06s/it] 20%|██        | 10/50 [1:45:51<7:02:45, 634.14s/it] 22%|██▏       | 11/50 [1:56:21<6:51:20, 632.83s/it]slurmstepd: error: *** JOB 42191 ON clr1247 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
