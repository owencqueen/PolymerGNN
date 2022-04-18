
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:31<8:35:20, 631.04s/it]  4%|▍         | 2/50 [21:02<8:25:01, 631.29s/it]  6%|▌         | 3/50 [31:29<8:13:26, 629.93s/it]  8%|▊         | 4/50 [41:59<8:02:58, 629.97s/it] 10%|█         | 5/50 [52:26<7:51:41, 628.92s/it] 12%|█▏        | 6/50 [1:02:55<7:41:20, 629.11s/it] 14%|█▍        | 7/50 [1:13:25<7:31:05, 629.43s/it] 16%|█▌        | 8/50 [1:23:58<7:21:15, 630.37s/it] 18%|█▊        | 9/50 [1:34:30<7:10:59, 630.73s/it] 20%|██        | 10/50 [1:44:58<7:00:04, 630.10s/it] 22%|██▏       | 11/50 [1:55:26<6:49:05, 629.37s/it]slurmstepd: error: *** JOB 42221 ON clr1247 CANCELLED AT 2022-04-14T16:20:46 DUE TO TIME LIMIT ***
