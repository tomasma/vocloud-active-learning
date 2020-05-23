# Starting a new project with a training set

Use this procedure when you wish to start a new project and you have a preprepared training set (e.g. a training set consisting of labelled spectra from Ondřejov collection).

- go to https://vocloud-dev.asu.cas.cz/
- select `Create job` from the main menu
- select `Standard jobs/Active learning`

The New Active learning job window should appear where you:
- fill in the `Project label` (any name, preferably your_project_name_iteration_number)
- in the `job configuration JSON` subwindow select `Use precreated configuration`
- select `learning_project_start_with_a_training_set.json`

In the JSON configuration preview panel below, a configuration suited for starting a new project (i.e. a configuration allowing you to build your initial training set) should appear. The individual configuration items are:
- `run_active_learning`: set it to "yes" if you wish to use active learning (neural network), if omitted or set otherwise, "labeling only" is started, this setting will eventually be omitted when the "Labeling" worker is finished
- `learning_session_name`: enter your project name, this is important as a folder of the same name gets created in the DATA/active-learning directory of the VO-CLOUD system (accessible by the Manage filesystem item in the main menu) where your training sets and labels, as well the performance statistics will be stored
- `classes`: enter the names of spectra classes that you want to distinguish (please use the preset format)
- `candidate_classes`: enter the names of spectra classes that you are interested in - these spectra will be stored in the system (please use the preset format)
- `iteration_num`: set it to 0 if you are starting a new project, increase it by 1 when you are ready for the next iteration (i.e. when you have saved results of the previous iteration)
- `training_set_csv`: set it to "/data/vocloud/filesystem/DATA/active-learning/training-set.csv" if you want to use preprepared labelled Ondřejov training set
- `pool_csv`: points to the file containing the spectra to be analyzed, in this example set to a csv file containing the LAMOST-DR2 collection of approximately 4 mil. spectra, incorrect path results in an error   
- `poolnames_csv`: points to the file containing the names of the spectra to be analyzed, in this example set to a csv file containing the LAMOST-DR2 collection of approximately 4 mil. spectra, incorrect path results in an error 
- `random_sample_size`: set it to 30 (number of random samples used to determine network performance)
- `batch_size`: set it in the range of 10-1000, recommended value: 100 (number of spectra with the highest uncertainty level that the user should label - will be added to the training set in the next iteration)

You most probably just need to set:
- `learning_session_name`
- `classes`
- `candidate_classes` (default: the last class)
