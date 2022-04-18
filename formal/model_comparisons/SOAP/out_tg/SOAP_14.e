
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:33<8:37:27, 633.62s/it]  4%|▍         | 2/50 [21:07<8:27:00, 633.76s/it]  6%|▌         | 3/50 [31:44<8:17:12, 634.73s/it]  8%|▊         | 4/50 [42:20<8:06:46, 634.92s/it] 10%|█         | 5/50 [52:55<7:56:16, 635.04s/it] 12%|█▏        | 6/50 [1:03:29<7:45:30, 634.78s/it] 14%|█▍        | 7/50 [1:14:07<7:35:40, 635.82s/it] 16%|█▌        | 8/50 [1:24:38<7:23:54, 634.16s/it] 18%|█▊        | 9/50 [1:35:09<7:12:52, 633.47s/it] 20%|██        | 10/50 [1:45:39<7:01:34, 632.37s/it] 22%|██▏       | 11/50 [1:56:12<6:51:09, 632.55s/it]slurmstepd: error: *** JOB 42193 ON clr0710 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
