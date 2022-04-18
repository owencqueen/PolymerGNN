
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:40<8:42:41, 640.03s/it]  4%|▍         | 2/50 [21:20<8:32:01, 640.04s/it]  6%|▌         | 3/50 [31:59<8:21:08, 639.76s/it]  8%|▊         | 4/50 [42:36<8:09:53, 638.99s/it] 10%|█         | 5/50 [53:18<7:59:50, 639.80s/it] 12%|█▏        | 6/50 [1:03:59<7:49:31, 640.27s/it] 14%|█▍        | 7/50 [1:14:36<7:38:06, 639.22s/it] 16%|█▌        | 8/50 [1:25:11<7:26:41, 638.13s/it] 18%|█▊        | 9/50 [1:35:46<7:15:15, 636.96s/it] 20%|██        | 10/50 [1:46:24<7:04:56, 637.42s/it] 22%|██▏       | 11/50 [1:57:00<6:53:59, 636.91s/it]slurmstepd: error: *** JOB 42199 ON clr0710 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
