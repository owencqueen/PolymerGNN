
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:34<8:38:28, 634.87s/it]  4%|▍         | 2/50 [21:07<8:27:25, 634.28s/it]  6%|▌         | 3/50 [31:39<8:16:12, 633.47s/it]  8%|▊         | 4/50 [42:15<8:06:18, 634.30s/it] 10%|█         | 5/50 [52:50<7:55:49, 634.43s/it] 12%|█▏        | 6/50 [1:03:28<7:46:01, 635.50s/it] 14%|█▍        | 7/50 [1:14:06<7:36:04, 636.38s/it] 16%|█▌        | 8/50 [1:24:39<7:24:37, 635.17s/it] 18%|█▊        | 9/50 [1:35:12<7:13:41, 634.66s/it] 20%|██        | 10/50 [1:45:46<7:02:57, 634.44s/it] 22%|██▏       | 11/50 [1:56:23<6:52:51, 635.17s/it]slurmstepd: error: *** JOB 42214 ON clr0738 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
