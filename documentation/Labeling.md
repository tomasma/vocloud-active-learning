There, you will see a list of spectra that, in general, belong to 3 categories:
- "oracle" set (white background color in the list): spectra that the convolutional neural network selected for expert evaluation because in case of these samples the network prediction had the highest level of uncertainty. Your labels will be stored and, in the next iteration, these samples will be added to the training set.  
- "perf-est" set (green background color in the list): spectra that were randomly selected for the purpose of estimating network performance.
- "candidates" (yellow background color in the list): spectra that the network selected as potentially interesting, i.e. belonging to the categories/classes of interest ("candidate classes"). These spectra will be displayed only in case your JSON configuration contained "show_candidates" option set to "yes". This is your potential result.

Now, your task is to label these spectra, i.e. to select the category they belong to. You may do this by clicking on the appropriate radio button or using a keyboard shortcut (`Alt + NumLock_number` corresponding to the selected category). Often, when proceeding fast, one mislabels a sample. In that case, you might use the `Previous` button or use `Alt + p` keyboard shortcut.

Typically, you will mainly decide based on the shape of the spectra in your wavelength range of interest (default: cca 652-673 nm). Occasionally, you might want to see the whole spectrum (press the `Raw spectra button`). 

Also, the metadata corresponding to the displayed sample are shown. The types of data that will be displayed might be changed by setting the "metadata2show" option in the job JSON configuration (this has to be set when you start a job). The default is set to "["filename","class","subclass","mag1","ra","dec","prediction","label","iteration","set"]".

You may also view metadata of all the samples. Simply press the `All metadata` button. Pressing the same button again will hide the table.

When you reach the end of the list, all the labels are automatically saved in the `DATA/active-learning/your_project` folder together with the performance estimation statistics. The statistics is also displayed in the top right corner of the display window. You may also download the labels and, optionally, also the spectra to your computer (hyperlinks labeled `Download labels` and `Download spectra`). The `Save` button allows to save the data at any time.

