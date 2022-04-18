
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:40<8:43:22, 640.87s/it]  4%|▍         | 2/50 [21:22<8:32:59, 641.24s/it]  6%|▌         | 3/50 [32:05<8:22:36, 641.64s/it]  8%|▊         | 4/50 [42:44<8:11:22, 640.93s/it] 10%|█         | 5/50 [53:21<7:59:48, 639.74s/it] 12%|█▏        | 6/50 [1:04:04<7:49:41, 640.50s/it] 14%|█▍        | 7/50 [1:14:43<7:38:48, 640.20s/it] 16%|█▌        | 8/50 [1:25:21<7:27:43, 639.62s/it] 18%|█▊        | 9/50 [1:35:59<7:16:46, 639.18s/it] 20%|██        | 10/50 [1:46:37<7:05:43, 638.58s/it] 22%|██▏       | 11/50 [1:57:17<6:55:21, 639.00s/it]slurmstepd: error: *** JOB 42205 ON clr0710 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
