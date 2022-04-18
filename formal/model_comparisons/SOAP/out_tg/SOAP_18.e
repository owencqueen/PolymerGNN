
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [10:38<8:41:37, 638.73s/it]  4%|▍         | 2/50 [21:16<8:30:40, 638.34s/it]  6%|▌         | 3/50 [31:52<8:19:38, 637.83s/it]  8%|▊         | 4/50 [42:31<8:09:05, 637.95s/it] 10%|█         | 5/50 [53:09<7:58:33, 638.08s/it] 12%|█▏        | 6/50 [1:03:46<7:47:44, 637.84s/it] 14%|█▍        | 7/50 [1:14:26<7:37:37, 638.54s/it] 16%|█▌        | 8/50 [1:25:05<7:26:58, 638.54s/it] 18%|█▊        | 9/50 [1:35:39<7:15:30, 637.33s/it] 20%|██        | 10/50 [1:46:15<7:04:26, 636.67s/it] 22%|██▏       | 11/50 [1:56:53<6:54:15, 637.33s/it]slurmstepd: error: *** JOB 42197 ON clr0710 CANCELLED AT 2022-04-14T16:17:16 DUE TO TIME LIMIT ***
