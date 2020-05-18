# vocloud-active-learning
Combination of Python, JavaScript/HTML, and Java scripts allowing VO-CLOUD system users to run active learning jobs.

The most important are:
  - `run-active-learning.py`: the main script containing the `run_active_learning(json_config_file)` function that parses the JSON configuration input file defining the parameters and switches provided by the user in the Active learning job VO-CLOUD interface
  - `activecnn.py`: the core `activeCnn` function runs the active learning convolution neural network iteration
  - `spectra_list.html.template`: JavaScript/HTML code defining web page display and possible user actions 
