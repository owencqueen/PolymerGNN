
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:37<8:40:34, 637.43s/it]  4%|▍         | 2/50 [21:09<8:28:45, 635.96s/it]  6%|▌         | 3/50 [31:40<8:16:58, 634.44s/it]  8%|▊         | 4/50 [42:11<8:05:33, 633.35s/it] 10%|█         | 5/50 [52:40<7:53:53, 631.86s/it] 12%|█▏        | 6/50 [1:03:09<7:42:51, 631.17s/it] 14%|█▍        | 7/50 [1:13:41<7:32:25, 631.29s/it] 16%|█▌        | 8/50 [1:24:15<7:22:27, 632.08s/it] 18%|█▊        | 9/50 [1:34:45<7:11:39, 631.70s/it] 20%|██        | 10/50 [1:45:15<7:00:46, 631.16s/it] 22%|██▏       | 11/50 [1:55:47<6:50:20, 631.30s/it]slurmstepd: error: *** JOB 42222 ON clr1247 CANCELLED AT 2022-04-14T16:20:46 DUE TO TIME LIMIT ***
