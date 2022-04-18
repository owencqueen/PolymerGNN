
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:34<8:38:27, 634.85s/it]  4%|▍         | 2/50 [21:10<8:28:10, 635.22s/it]  6%|▌         | 3/50 [31:40<8:16:19, 633.61s/it]  8%|▊         | 4/50 [42:09<8:04:41, 632.21s/it] 10%|█         | 5/50 [52:39<7:53:41, 631.59s/it] 12%|█▏        | 6/50 [1:03:11<7:43:16, 631.74s/it] 14%|█▍        | 7/50 [1:13:44<7:32:56, 632.02s/it] 16%|█▌        | 8/50 [1:24:14<7:21:55, 631.32s/it] 18%|█▊        | 9/50 [1:34:42<7:10:39, 630.24s/it] 20%|██        | 10/50 [1:45:13<7:00:28, 630.72s/it] 22%|██▏       | 11/50 [1:55:46<6:50:25, 631.42s/it]slurmstepd: error: *** JOB 42212 ON clr0738 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
