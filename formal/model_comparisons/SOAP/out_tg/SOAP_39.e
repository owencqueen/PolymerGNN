
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:39<8:42:16, 639.51s/it]  4%|▍         | 2/50 [21:14<8:30:32, 638.17s/it]  6%|▌         | 3/50 [31:48<8:18:47, 636.77s/it]  8%|▊         | 4/50 [42:31<8:09:48, 638.87s/it] 10%|█         | 5/50 [53:11<7:59:22, 639.17s/it] 12%|█▏        | 6/50 [1:03:45<7:47:30, 637.51s/it] 14%|█▍        | 7/50 [1:14:26<7:37:38, 638.56s/it] 16%|█▌        | 8/50 [1:25:07<7:27:27, 639.22s/it] 18%|█▊        | 9/50 [1:35:45<7:16:32, 638.84s/it] 20%|██        | 10/50 [1:46:21<7:05:30, 638.27s/it] 22%|██▏       | 11/50 [1:56:59<6:54:39, 637.94s/it]slurmstepd: error: *** JOB 42218 ON clr0738 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
