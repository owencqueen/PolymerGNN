
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:34<8:37:55, 634.19s/it]  4%|▍         | 2/50 [21:09<8:27:38, 634.56s/it]  6%|▌         | 3/50 [31:45<8:17:27, 635.05s/it]  8%|▊         | 4/50 [42:19<8:06:35, 634.69s/it] 10%|█         | 5/50 [52:56<7:56:24, 635.22s/it] 12%|█▏        | 6/50 [1:03:27<7:44:57, 634.04s/it] 14%|█▍        | 7/50 [1:14:00<7:34:11, 633.77s/it] 16%|█▌        | 8/50 [1:24:35<7:23:48, 634.02s/it] 18%|█▊        | 9/50 [1:35:07<7:12:51, 633.45s/it] 20%|██        | 10/50 [1:45:43<7:02:53, 634.34s/it] 22%|██▏       | 11/50 [1:56:16<6:51:56, 633.77s/it]slurmstepd: error: *** JOB 42226 ON clr1247 CANCELLED AT 2022-04-14T16:25:16 DUE TO TIME LIMIT ***
