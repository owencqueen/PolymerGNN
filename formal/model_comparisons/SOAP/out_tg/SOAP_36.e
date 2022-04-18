
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:41<8:43:57, 641.58s/it]  4%|▍         | 2/50 [21:22<8:33:04, 641.35s/it]  6%|▌         | 3/50 [32:05<8:22:49, 641.90s/it]  8%|▊         | 4/50 [42:45<8:11:40, 641.31s/it] 10%|█         | 5/50 [53:28<8:01:23, 641.86s/it] 12%|█▏        | 6/50 [1:04:06<7:49:49, 640.68s/it] 14%|█▍        | 7/50 [1:14:47<7:39:16, 640.85s/it] 16%|█▌        | 8/50 [1:25:30<7:28:52, 641.26s/it] 18%|█▊        | 9/50 [1:36:12<7:18:20, 641.47s/it] 20%|██        | 10/50 [1:46:51<7:07:10, 640.77s/it] 22%|██▏       | 11/50 [1:57:31<6:56:26, 640.69s/it]slurmstepd: error: *** JOB 42215 ON clr0738 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
