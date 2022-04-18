
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [09:52<8:04:05, 592.76s/it]  4%|▍         | 2/50 [19:29<7:50:14, 587.81s/it]  6%|▌         | 3/50 [28:48<7:33:43, 579.22s/it]  8%|▊         | 4/50 [38:08<7:19:49, 573.69s/it] 10%|█         | 5/50 [47:42<7:10:07, 573.50s/it] 12%|█▏        | 6/50 [57:30<7:03:46, 577.88s/it] 14%|█▍        | 7/50 [1:07:25<6:57:52, 583.07s/it] 16%|█▌        | 8/50 [1:17:31<6:52:53, 589.86s/it] 18%|█▊        | 9/50 [1:27:31<6:45:19, 593.15s/it] 20%|██        | 10/50 [1:37:33<6:37:09, 595.73s/it] 22%|██▏       | 11/50 [1:47:35<6:28:29, 597.68s/it] 24%|██▍       | 12/50 [1:57:35<6:18:59, 598.41s/it]slurmstepd: error: *** JOB 42228 ON clr0708 CANCELLED AT 2022-04-14T16:28:46 DUE TO TIME LIMIT ***
