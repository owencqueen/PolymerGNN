
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:31<8:35:33, 631.30s/it]  4%|▍         | 2/50 [20:59<8:24:24, 630.50s/it]  6%|▌         | 3/50 [31:33<8:14:41, 631.52s/it]  8%|▊         | 4/50 [42:03<8:03:38, 630.83s/it] 10%|█         | 5/50 [52:30<7:52:26, 629.92s/it] 12%|█▏        | 6/50 [1:03:02<7:42:22, 630.51s/it] 14%|█▍        | 7/50 [1:13:36<7:32:32, 631.46s/it] 16%|█▌        | 8/50 [1:24:08<7:22:07, 631.60s/it] 18%|█▊        | 9/50 [1:34:40<7:11:41, 631.75s/it] 20%|██        | 10/50 [1:45:10<7:00:49, 631.23s/it] 22%|██▏       | 11/50 [1:55:43<6:50:34, 631.65s/it]slurmstepd: error: *** JOB 42190 ON clr1247 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
