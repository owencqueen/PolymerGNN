
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:34<8:38:23, 634.77s/it]  4%|▍         | 2/50 [21:08<8:27:37, 634.52s/it]  6%|▌         | 3/50 [31:41<8:16:36, 633.97s/it]  8%|▊         | 4/50 [42:11<8:05:05, 632.73s/it] 10%|█         | 5/50 [52:43<7:54:24, 632.54s/it] 12%|█▏        | 6/50 [1:03:13<7:43:23, 631.90s/it] 14%|█▍        | 7/50 [1:13:45<7:32:45, 631.77s/it] 16%|█▌        | 8/50 [1:24:17<7:22:25, 632.04s/it] 18%|█▊        | 9/50 [1:34:49<7:11:50, 631.96s/it] 20%|██        | 10/50 [1:45:18<7:00:46, 631.16s/it] 22%|██▏       | 11/50 [1:55:48<6:50:02, 630.83s/it]slurmstepd: error: *** JOB 42203 ON clr0710 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
