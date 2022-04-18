
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:29<8:34:26, 629.93s/it]  4%|▍         | 2/50 [21:03<8:24:45, 630.96s/it]  6%|▌         | 3/50 [31:35<8:14:34, 631.37s/it]  8%|▊         | 4/50 [42:08<8:04:28, 631.92s/it] 10%|█         | 5/50 [52:40<7:53:53, 631.86s/it] 12%|█▏        | 6/50 [1:03:10<7:42:54, 631.24s/it] 14%|█▍        | 7/50 [1:13:43<7:32:53, 631.94s/it] 16%|█▌        | 8/50 [1:24:15<7:22:15, 631.80s/it] 18%|█▊        | 9/50 [1:34:45<7:11:23, 631.30s/it] 20%|██        | 10/50 [1:45:17<7:00:55, 631.39s/it] 22%|██▏       | 11/50 [1:55:50<6:50:49, 632.04s/it]slurmstepd: error: *** JOB 42202 ON clr0738 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
