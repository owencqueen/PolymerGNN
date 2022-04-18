
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:31<8:35:51, 631.66s/it]  4%|▍         | 2/50 [21:00<8:24:34, 630.71s/it]  6%|▌         | 3/50 [31:31<8:14:10, 630.85s/it]  8%|▊         | 4/50 [42:03<8:03:56, 631.23s/it] 10%|█         | 5/50 [52:33<7:53:08, 630.86s/it] 12%|█▏        | 6/50 [1:03:03<7:42:28, 630.65s/it] 14%|█▍        | 7/50 [1:13:36<7:32:25, 631.28s/it] 16%|█▌        | 8/50 [1:24:06<7:21:34, 630.83s/it] 18%|█▊        | 9/50 [1:34:37<7:11:11, 631.01s/it] 20%|██        | 10/50 [1:45:11<7:01:13, 631.84s/it] 22%|██▏       | 11/50 [1:55:42<6:50:29, 631.53s/it]slurmstepd: error: *** JOB 42192 ON clr0823 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
