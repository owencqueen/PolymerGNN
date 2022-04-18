
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
  0%|          | 0/50 [00:00<?, ?it/s]  2%|▏         | 1/50 [09:43<7:56:45, 583.78s/it]  4%|▍         | 2/50 [19:06<7:42:04, 577.60s/it]  6%|▌         | 3/50 [28:29<7:28:52, 573.03s/it]  8%|▊         | 4/50 [37:53<7:17:20, 570.43s/it] 10%|█         | 5/50 [47:33<7:09:50, 573.12s/it]slurmstepd: error: *** JOB 41995 ON clr1247 CANCELLED AT 2022-04-14T14:08:25 ***
