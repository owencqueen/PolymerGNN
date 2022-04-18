
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:36<8:39:39, 636.31s/it]  4%|▍         | 2/50 [21:13<8:29:14, 636.54s/it]  6%|▌         | 3/50 [31:51<8:18:57, 636.97s/it]  8%|▊         | 4/50 [42:26<8:07:53, 636.38s/it] 10%|█         | 5/50 [53:05<7:57:57, 637.27s/it] 12%|█▏        | 6/50 [1:03:42<7:47:18, 637.23s/it] 14%|█▍        | 7/50 [1:14:18<7:36:19, 636.73s/it] 16%|█▌        | 8/50 [1:24:57<7:26:08, 637.34s/it] 18%|█▊        | 9/50 [1:35:33<7:15:24, 637.18s/it] 20%|██        | 10/50 [1:46:09<7:04:26, 636.66s/it] 22%|██▏       | 11/50 [1:56:48<6:54:14, 637.30s/it]slurmstepd: error: *** JOB 42216 ON clr0738 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
