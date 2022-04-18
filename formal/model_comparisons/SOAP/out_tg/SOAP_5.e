
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:10<8:18:57, 610.98s/it]  4%|▍         | 2/50 [20:21<8:08:33, 610.69s/it]  6%|▌         | 3/50 [30:33<7:58:44, 611.16s/it]  8%|▊         | 4/50 [40:42<7:48:11, 610.68s/it] 10%|█         | 5/50 [50:54<7:38:15, 611.02s/it] 12%|█▏        | 6/50 [1:01:04<7:27:45, 610.59s/it] 14%|█▍        | 7/50 [1:11:07<7:16:04, 608.48s/it] 16%|█▌        | 8/50 [1:21:20<7:06:46, 609.67s/it] 18%|█▊        | 9/50 [1:31:38<6:58:16, 612.12s/it] 20%|██        | 10/50 [1:41:55<6:49:04, 613.61s/it] 22%|██▏       | 11/50 [1:52:12<6:39:38, 614.83s/it]slurmstepd: error: *** JOB 42184 ON clr0724 CANCELLED AT 2022-04-14T16:16:46 DUE TO TIME LIMIT ***
