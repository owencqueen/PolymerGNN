
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:39<8:42:03, 639.25s/it]  4%|▍         | 2/50 [21:16<8:31:01, 638.78s/it]  6%|▌         | 3/50 [31:53<8:19:56, 638.23s/it]  8%|▊         | 4/50 [42:29<8:08:38, 637.36s/it] 10%|█         | 5/50 [53:02<7:57:05, 636.12s/it] 12%|█▏        | 6/50 [1:03:38<7:46:33, 636.21s/it] 14%|█▍        | 7/50 [1:14:13<7:35:41, 635.85s/it] 16%|█▌        | 8/50 [1:24:50<7:25:11, 635.99s/it] 18%|█▊        | 9/50 [1:35:24<7:14:16, 635.53s/it] 20%|██        | 10/50 [1:45:58<7:03:16, 634.92s/it] 22%|██▏       | 11/50 [1:56:34<6:52:57, 635.31s/it]slurmstepd: error: *** JOB 42213 ON clr0738 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
