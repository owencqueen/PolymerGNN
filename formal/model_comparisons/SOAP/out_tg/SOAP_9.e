
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:19<8:25:45, 619.29s/it]  4%|▍         | 2/50 [20:15<8:09:51, 612.32s/it]  6%|▌         | 3/50 [29:56<7:52:17, 602.93s/it]  8%|▊         | 4/50 [39:17<7:32:32, 590.27s/it] 10%|█         | 5/50 [48:43<7:17:23, 583.20s/it] 12%|█▏        | 6/50 [58:18<7:05:43, 580.53s/it] 14%|█▍        | 7/50 [1:08:05<6:57:32, 582.61s/it] 16%|█▌        | 8/50 [1:18:00<6:50:21, 586.23s/it] 18%|█▊        | 9/50 [1:28:04<6:44:14, 591.57s/it] 20%|██        | 10/50 [1:38:05<6:36:19, 594.49s/it] 22%|██▏       | 11/50 [1:48:09<6:28:13, 597.27s/it] 24%|██▍       | 12/50 [1:58:17<6:20:22, 600.60s/it]slurmstepd: error: *** JOB 42188 ON clr0708 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
