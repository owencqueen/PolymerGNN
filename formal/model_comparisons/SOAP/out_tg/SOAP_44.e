
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:31<8:35:29, 631.22s/it]  4%|▍         | 2/50 [21:00<8:24:24, 630.52s/it]  6%|▌         | 3/50 [31:31<8:14:05, 630.75s/it]  8%|▊         | 4/50 [42:03<8:03:54, 631.17s/it] 10%|█         | 5/50 [52:37<7:53:57, 631.94s/it] 12%|█▏        | 6/50 [1:03:14<7:44:33, 633.49s/it] 14%|█▍        | 7/50 [1:13:44<7:33:15, 632.45s/it] 16%|█▌        | 8/50 [1:24:14<7:22:19, 631.89s/it] 18%|█▊        | 9/50 [1:34:47<7:11:57, 632.14s/it] 20%|██        | 10/50 [1:45:19<7:01:18, 631.97s/it] 22%|██▏       | 11/50 [1:55:51<6:50:46, 631.95s/it]slurmstepd: error: *** JOB 42223 ON clr1247 CANCELLED AT 2022-04-14T16:20:46 DUE TO TIME LIMIT ***
