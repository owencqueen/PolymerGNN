
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:18<8:24:55, 618.27s/it]  4%|▍         | 2/50 [20:30<8:13:14, 616.56s/it]  6%|▌         | 3/50 [30:43<8:02:06, 615.46s/it]  8%|▊         | 4/50 [40:56<7:51:13, 614.64s/it] 10%|█         | 5/50 [51:07<7:40:11, 613.60s/it] 12%|█▏        | 6/50 [1:01:17<7:29:03, 612.35s/it] 14%|█▍        | 7/50 [1:11:26<7:18:09, 611.38s/it] 16%|█▌        | 8/50 [1:21:42<7:09:05, 612.98s/it] 18%|█▊        | 9/50 [1:32:03<7:00:23, 615.20s/it] 20%|██        | 10/50 [1:42:23<6:51:08, 616.71s/it] 22%|██▏       | 11/50 [1:52:40<6:40:52, 616.73s/it]slurmstepd: error: *** JOB 42186 ON clr0724 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
