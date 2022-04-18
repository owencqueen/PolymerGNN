
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:34<8:37:58, 634.25s/it]  4%|▍         | 2/50 [21:05<8:26:45, 633.44s/it]  6%|▌         | 3/50 [31:38<8:16:07, 633.36s/it]  8%|▊         | 4/50 [42:13<8:05:53, 633.78s/it] 10%|█         | 5/50 [52:45<7:54:54, 633.21s/it] 12%|█▏        | 6/50 [1:03:18<7:44:21, 633.21s/it] 14%|█▍        | 7/50 [1:13:53<7:34:03, 633.57s/it] 16%|█▌        | 8/50 [1:24:28<7:23:51, 634.09s/it] 18%|█▊        | 9/50 [1:35:04<7:13:46, 634.80s/it] 20%|██        | 10/50 [1:45:41<7:03:34, 635.36s/it] 22%|██▏       | 11/50 [1:56:14<6:52:27, 634.55s/it]slurmstepd: error: *** JOB 42206 ON clr0823 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
