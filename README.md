# vocloud-active-learning
Combination of Python, JavaScript/HTML, and Java scripts allowing VO-CLOUD system users to run active learning jobs.

The most important are:
  - `run-active-learning.py`: the main script containing the `run_active_learning(json_config_file)` function that parses the JSON configuration input file defining the parameters and switches provided by the user in the Active learning job VO-CLOUD interface
  - `activecnn.py`: the core `activeCnn` function that runs the active learning convolution neural network iteration
  - `data_handler.py`: contains support functions for handling spectral data and metadata
  - `spectra_list.html.template`: JavaScript/HTML code defining web page display and possible user actions 
  - `LabelDataUpload.java`: Java script allowing to transfer user-provided data labels to the VO-CLOUD system 
  
# How to use it?

In case you do not have an already prepared a training set, use the following protocol: 

[Start a new project](./documentation/Starting_a_new_project.md)

In case you have already prepared a training set (e.g. labelled spectra from Ondřejov dataset), you should use the following procedure instead:

[Start a new project with a training set](./documentation/Starting_a_new_project_with_a_training_set.md)

Continue with the iterative active learning process: 

[Continue learning](./documentation/Continue_learning.md)

If you want to see the results:

[See the results](./documentation/See_the_results.md)
