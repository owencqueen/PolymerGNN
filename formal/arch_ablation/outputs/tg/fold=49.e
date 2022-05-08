
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


Traceback (most recent call last):
  File "src/mono_graph.py", line 6, in <module>
    from polymerlearn.utils import get_IV_add, get_Tg_add, GraphDataset
  File "/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/utils/__init__.py", line 1, in <module>
    from .graph_prep import GraphDataset
  File "/lustre/isaac/scratch/oqueen/PolymerGNN/polymerlearn/utils/graph_prep.py", line 216
    )
    ^
SyntaxError: invalid syntax
