# vocloud-active-learning
Combination of Python, JavaScript/HTML, and Java scripts allowing VO-CLOUD system users to run active learning jobs.

The most important are:
  - `run-active-learning.py`: the main script containing the `run_active_learning(json_config_file)` function that parses the JSON configuration input file defining the parameters and switches provided by the user in the Active learning job VO-CLOUD interface
  - `activecnn.py`: the core `activeCnn` function that runs the active learning convolution neural network iteration
  - `data_handler.py`: contains support functions for handling spectral data and metadata
  - `spectra_list.html.template`: JavaScript/HTML code defining web page display and possible user actions 
  - `LabelDataUpload.java`: Java script allowing to transfer user-provided data labels to the VO-CLOUD system 
  
# How to use it?

# Starting a new project

- go to https://vocloud-dev.asu.cas.cz/
- select Create job from the main menu
- select Standard jobs/Active learning

The New Active learning job window should appear where you:
- fill in the Project label (any name)
- in the job configuration JSON subwindow select Use precreated configuration
- select active_learning_iteration0.json

In the JSON configuration preview panel below a configuration suited for starting a new project (i.e. a configuration allowing you to build your initial training set) should appear. The individual configuration items are:
- run_active_learning: set it to "yes" if you wish to use active learning (neural network), if omitted or set tootherwise, :labeling only" is started, this setting will eventually be omitted when the "Labeling" worker gets set
- learning_session_name: enter your project name, this is important as a folder of the same name gets created in the DATA/active-learning directory of the VO-CLOUD system (accessible by the Manage filesystem item in the main menu) where your training sets and labels, as well the performance statistics will be stored
- classes: enter the names of spectra classes that you want to distinguish in the preset format
- candidate_classes: enter the names of spectra classes that you are interested in - these spectra will be stored in the system
- iteration_num: set it to 0 if you are starting a new project, increase it by 1 when you are ready for the next iteration (i.e. when you have saved results of the previous iteration)
- pool_csv: points to the file containing the spectra to be analyzed, in this example set to a csv file containing the LAMOST-DR2 collection of approximately 4 mil. spectra, incorrect path results in an error   
- poolnames_csv: points to the file containing the names of the spectra to be analyzed, in this example set to a csv file containing the LAMOST-DR2 collection of approximately 4 mil. spectra, incorrect path results in an error 
- random_sample_size: set it to 0 if you are starting a new project (iteration 0), for higher iterations, the recommended setting is 30 (number of random samples used to determine network performance)
- batch_size: set it in the range of 100-500 if you are starting a new project (iteration 0, note: the initial training set has to contain at least 6 examples of each class, high numbers make the labeling session tedious), for higher iterations, the value of 100 is recommended (number of spectra with the highest uncertainty level that the user should label - will be added to the training set in the next iteration)

You most probably just need to set:
- learning_session_name
- classes
- candidate_classes (default: the last class)
- batch_size: set it to a lower number if the frequency of your classes is more even, leave it high if any of the classes is rare (remember, you need 6 examples of each class in the initial training set)
