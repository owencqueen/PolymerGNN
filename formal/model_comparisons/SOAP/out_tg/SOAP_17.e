
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:33<8:37:23, 633.53s/it]  4%|▍         | 2/50 [21:06<8:26:41, 633.36s/it]  6%|▌         | 3/50 [31:39<8:16:08, 633.38s/it]  8%|▊         | 4/50 [42:11<8:05:08, 632.79s/it] 10%|█         | 5/50 [52:46<7:55:02, 633.39s/it] 12%|█▏        | 6/50 [1:03:16<7:43:55, 632.63s/it] 14%|█▍        | 7/50 [1:13:49<7:33:17, 632.50s/it] 16%|█▌        | 8/50 [1:24:17<7:21:51, 631.22s/it] 18%|█▊        | 9/50 [1:34:51<7:11:54, 632.06s/it] 20%|██        | 10/50 [1:45:22<7:01:13, 631.85s/it] 22%|██▏       | 11/50 [1:55:52<6:50:13, 631.11s/it]slurmstepd: error: *** JOB 42196 ON clr0710 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
