
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:42<8:44:25, 642.15s/it]  4%|▍         | 2/50 [21:23<8:33:35, 642.00s/it]  6%|▌         | 3/50 [32:00<8:21:39, 640.42s/it]  8%|▊         | 4/50 [42:38<8:10:30, 639.79s/it] 10%|█         | 5/50 [53:17<7:59:31, 639.38s/it] 12%|█▏        | 6/50 [1:03:59<7:49:24, 640.11s/it] 14%|█▍        | 7/50 [1:14:41<7:39:11, 640.73s/it] 16%|█▌        | 8/50 [1:25:24<7:28:55, 641.33s/it] 18%|█▊        | 9/50 [1:36:04<7:18:00, 641.00s/it] 20%|██        | 10/50 [1:46:43<7:07:00, 640.52s/it] 22%|██▏       | 11/50 [1:57:24<6:56:26, 640.68s/it]slurmstepd: error: *** JOB 42207 ON clr0738 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
