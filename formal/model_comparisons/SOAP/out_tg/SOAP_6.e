
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:12<8:19:59, 612.24s/it]  4%|▍         | 2/50 [20:27<8:10:35, 613.23s/it]  6%|▌         | 3/50 [30:40<8:00:19, 613.18s/it]  8%|▊         | 4/50 [40:53<7:49:55, 612.94s/it] 10%|█         | 5/50 [51:06<7:39:49, 613.10s/it] 12%|█▏        | 6/50 [1:01:15<7:28:45, 611.94s/it] 14%|█▍        | 7/50 [1:11:26<7:18:10, 611.41s/it] 16%|█▌        | 8/50 [1:21:41<7:08:46, 612.53s/it] 18%|█▊        | 9/50 [1:32:01<7:00:06, 614.79s/it] 20%|██        | 10/50 [1:42:16<6:49:56, 614.90s/it] 22%|██▏       | 11/50 [1:52:35<6:40:24, 616.01s/it]slurmstepd: error: *** JOB 42185 ON clr0724 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
