
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:38<8:41:21, 638.39s/it]  4%|▍         | 2/50 [21:17<8:30:50, 638.55s/it]  6%|▌         | 3/50 [31:56<8:20:19, 638.71s/it]  8%|▊         | 4/50 [42:32<8:09:02, 637.88s/it] 10%|█         | 5/50 [53:11<7:58:42, 638.27s/it] 12%|█▏        | 6/50 [1:03:53<7:48:48, 639.30s/it] 14%|█▍        | 7/50 [1:14:33<7:38:27, 639.71s/it] 16%|█▌        | 8/50 [1:25:11<7:27:22, 639.11s/it] 18%|█▊        | 9/50 [1:35:52<7:17:11, 639.80s/it] 20%|██        | 10/50 [1:46:29<7:05:46, 638.67s/it] 22%|██▏       | 11/50 [1:57:05<6:54:42, 638.01s/it]slurmstepd: error: *** JOB 42210 ON clr0738 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
