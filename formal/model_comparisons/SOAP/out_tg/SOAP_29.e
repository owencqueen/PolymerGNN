
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:39<8:42:27, 639.75s/it]  4%|▍         | 2/50 [21:17<8:31:17, 639.12s/it]  6%|▌         | 3/50 [31:55<8:20:31, 638.96s/it]  8%|▊         | 4/50 [42:36<8:10:10, 639.35s/it] 10%|█         | 5/50 [53:12<7:58:52, 638.50s/it] 12%|█▏        | 6/50 [1:03:49<7:47:56, 638.10s/it] 14%|█▍        | 7/50 [1:14:27<7:37:11, 637.95s/it] 16%|█▌        | 8/50 [1:25:05<7:26:35, 637.99s/it] 18%|█▊        | 9/50 [1:35:44<7:16:09, 638.28s/it] 20%|██        | 10/50 [1:46:20<7:05:01, 637.54s/it] 22%|██▏       | 11/50 [1:56:57<6:54:24, 637.55s/it]slurmstepd: error: *** JOB 42208 ON clr0710 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
