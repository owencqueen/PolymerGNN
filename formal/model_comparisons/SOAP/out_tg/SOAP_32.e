
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:33<8:37:19, 633.46s/it]  4%|▍         | 2/50 [21:07<8:27:00, 633.75s/it]  6%|▌         | 3/50 [31:45<8:17:18, 634.87s/it]  8%|▊         | 4/50 [42:20<8:06:45, 634.89s/it] 10%|█         | 5/50 [52:56<7:56:26, 635.25s/it] 12%|█▏        | 6/50 [1:03:31<7:45:50, 635.24s/it] 14%|█▍        | 7/50 [1:14:11<7:36:13, 636.59s/it] 16%|█▌        | 8/50 [1:24:43<7:24:37, 635.17s/it] 18%|█▊        | 9/50 [1:35:16<7:13:43, 634.73s/it] 20%|██        | 10/50 [1:45:49<7:02:46, 634.16s/it] 22%|██▏       | 11/50 [1:56:20<6:51:29, 633.05s/it]slurmstepd: error: *** JOB 42211 ON clr0738 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
