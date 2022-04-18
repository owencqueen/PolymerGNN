
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:37<8:40:55, 637.86s/it]  4%|▍         | 2/50 [21:09<8:28:43, 635.91s/it]  6%|▌         | 3/50 [31:42<8:17:25, 635.01s/it]  8%|▊         | 4/50 [42:14<8:06:08, 634.10s/it] 10%|█         | 5/50 [52:48<7:55:36, 634.14s/it] 12%|█▏        | 6/50 [1:03:33<7:47:28, 637.46s/it] 14%|█▍        | 7/50 [1:14:20<7:38:58, 640.44s/it] 16%|█▌        | 8/50 [1:25:06<7:29:18, 641.88s/it] 18%|█▊        | 9/50 [1:35:41<7:17:16, 639.92s/it] 20%|██        | 10/50 [1:46:12<7:04:51, 637.28s/it] 22%|██▏       | 11/50 [1:56:43<6:53:03, 635.47s/it]slurmstepd: error: *** JOB 42204 ON clr0710 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
