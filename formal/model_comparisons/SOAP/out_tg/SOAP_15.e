
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:32<8:36:34, 632.54s/it]  4%|▍         | 2/50 [21:02<8:25:28, 631.84s/it]  6%|▌         | 3/50 [31:39<8:16:06, 633.33s/it]  8%|▊         | 4/50 [42:11<8:05:17, 633.00s/it] 10%|█         | 5/50 [52:44<7:54:43, 632.98s/it] 12%|█▏        | 6/50 [1:03:17<7:44:02, 632.79s/it] 14%|█▍        | 7/50 [1:13:51<7:33:46, 633.18s/it] 16%|█▌        | 8/50 [1:24:21<7:22:39, 632.38s/it] 18%|█▊        | 9/50 [1:34:52<7:11:42, 631.78s/it] 20%|██        | 10/50 [1:45:24<7:01:20, 632.02s/it] 22%|██▏       | 11/50 [1:55:58<6:51:08, 632.51s/it]slurmstepd: error: *** JOB 42194 ON clr1247 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
