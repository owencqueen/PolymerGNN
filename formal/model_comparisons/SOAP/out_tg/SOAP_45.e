
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:34<8:37:47, 634.04s/it]  4%|▍         | 2/50 [21:06<8:26:46, 633.47s/it]  6%|▌         | 3/50 [31:41<8:16:38, 634.01s/it]  8%|▊         | 4/50 [42:13<8:05:34, 633.37s/it] 10%|█         | 5/50 [52:45<7:54:45, 633.02s/it] 12%|█▏        | 6/50 [1:03:19<7:44:21, 633.22s/it] 14%|█▍        | 7/50 [1:13:51<7:33:41, 633.06s/it] 16%|█▌        | 8/50 [1:24:25<7:23:18, 633.30s/it] 18%|█▊        | 9/50 [1:34:57<7:12:26, 632.85s/it] 20%|██        | 10/50 [1:45:31<7:02:12, 633.32s/it] 22%|██▏       | 11/50 [1:56:06<6:51:54, 633.69s/it]slurmstepd: error: *** JOB 42224 ON clr1247 CANCELLED AT 2022-04-14T16:20:46 DUE TO TIME LIMIT ***
