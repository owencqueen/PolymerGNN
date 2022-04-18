
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:31<8:36:07, 631.99s/it]  4%|▍         | 2/50 [21:00<8:24:45, 630.95s/it]  6%|▌         | 3/50 [31:35<8:15:13, 632.20s/it]  8%|▊         | 4/50 [42:14<8:06:17, 634.30s/it] 10%|█         | 5/50 [52:52<7:56:31, 635.37s/it] 12%|█▏        | 6/50 [1:03:30<7:46:22, 635.96s/it] 14%|█▍        | 7/50 [1:14:10<7:36:50, 637.45s/it] 16%|█▌        | 8/50 [1:24:46<7:25:50, 636.92s/it] 18%|█▊        | 9/50 [1:35:26<7:15:45, 637.71s/it] 20%|██        | 10/50 [1:46:04<7:05:12, 637.80s/it] 22%|██▏       | 11/50 [1:56:42<6:54:34, 637.81s/it]slurmstepd: error: *** JOB 42209 ON clr0738 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
