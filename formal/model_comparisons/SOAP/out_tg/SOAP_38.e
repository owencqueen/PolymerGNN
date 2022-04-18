
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:37<8:40:14, 637.03s/it]  4%|▍         | 2/50 [21:16<8:30:16, 637.84s/it]  6%|▌         | 3/50 [31:51<8:19:00, 637.04s/it]  8%|▊         | 4/50 [42:26<8:07:46, 636.23s/it] 10%|█         | 5/50 [53:01<7:56:51, 635.81s/it] 12%|█▏        | 6/50 [1:03:38<7:46:35, 636.27s/it] 14%|█▍        | 7/50 [1:14:12<7:35:28, 635.55s/it] 16%|█▌        | 8/50 [1:24:46<7:24:34, 635.10s/it] 18%|█▊        | 9/50 [1:35:21<7:13:59, 635.11s/it] 20%|██        | 10/50 [1:45:59<7:04:02, 636.05s/it] 22%|██▏       | 11/50 [1:56:35<6:53:21, 635.94s/it]slurmstepd: error: *** JOB 42217 ON clr0710 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
