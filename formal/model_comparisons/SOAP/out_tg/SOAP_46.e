
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:32<8:36:34, 632.54s/it]  4%|▍         | 2/50 [21:01<8:25:13, 631.53s/it]  6%|▌         | 3/50 [31:32<8:14:24, 631.16s/it]  8%|▊         | 4/50 [42:02<8:03:48, 631.06s/it] 10%|█         | 5/50 [52:31<7:52:51, 630.47s/it] 12%|█▏        | 6/50 [1:03:00<7:41:53, 629.85s/it] 14%|█▍        | 7/50 [1:13:32<7:31:48, 630.42s/it] 16%|█▌        | 8/50 [1:24:00<7:20:52, 629.81s/it] 18%|█▊        | 9/50 [1:34:27<7:09:51, 629.07s/it] 20%|██        | 10/50 [1:44:57<6:59:25, 629.15s/it] 22%|██▏       | 11/50 [1:55:27<6:49:08, 629.45s/it]slurmstepd: error: *** JOB 42225 ON clr1247 CANCELLED AT 2022-04-14T16:25:16 DUE TO TIME LIMIT ***
