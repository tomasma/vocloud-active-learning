import sys
import json
import string
import os
import csv

from datetime import datetime
from elasticsearch import Elasticsearch

import pandas as pd
import numpy as np
from scipy.stats import entropy
from astropy.io.votable.tree import VOTableFile, Resource, Table, Field

import base.data_handler as dh

from active_cnn import data
from active_cnn import model
from active_cnn import preprocessing
from activecnn import activeCnn
import matplotlib.pyplot as plt

__LINK_TEMPLATE = string.Template('<option selected id="${spectrum_name}_link" style="background-color:${background_color};">${spectrum_name}</option>\n')
__script_dir = os.path.dirname(os.path.realpath(__file__))


def main():
    if len(sys.argv) < 2:
        raise ValueError('Must pass at least one input file')
    for json_config_file in sys.argv[1:]:
        run_active_learning(json_config_file)

def _generate_spectra(folder_prefix,folder,spectra,database_index,metadata_df,classes,prediction_file,random_sample_size,oracle_size,cat_list,metadata2show,iteration_num,spectra2add_fname,labels2add_fname,raw_spectra_source,statistics,learning_session_name):
    with open(__script_dir + "/spectra_list.html.template") as template_file:
        html_template = string.Template(template_file.read())
    n_display_limit = 1000
    spectra_list = []
    metadata = []
    fname_list = []
    prediction_list = []
    entropy_list = []
    label_list = []
    oracle_perfest_list=[]
    comment_list = []
    background_color = []
    prediction=[]
    entropy = -1
    try:
       f=open(prediction_file,'r')
       b=f.readline()
       while len(b)>3:
          b_array = b.split(',')
          if len(b_array) > 2:
             entropy = float(b_array[2])
          else:
             entropy = -1
          if len(b_array) > 1:
             prediction_num = int(b_array[1])
             prediction = cat_list[prediction_num]
          else:
             prediction = ""
          prediction_list.append(prediction)
          entropy_list.append(round(float(entropy),5))
          b=f.readline()
       f.close()
    except:
       entropy_list.append(entropy)
       prediction_list.append(prediction)

    es=Elasticsearch([{'host':'localhost','port':9200}])
    i = 0
    background_color.append('white')
    for idx, spectrum in spectra.iterrows():
        spectrum_link = __LINK_TEMPLATE.substitute({'spectrum_name': str(idx),
                                                    'spectrum_name_short': str(idx),
                                                    'background_color': background_color[i]})
        spectra_list.append(spectrum_link)
        metadata_row = []
        fname = []
        comment = []
        try:
             prediction=prediction_list[i]
        except:
             prediction=[]
        try:
             entropy=entropy_list[i]
        except:
             entropy=[]
        label=[]
        sample2label_size = oracle_size + random_sample_size
        if oracle_size + random_sample_size > 0: 
           if i < oracle_size:
               oracle_perfest = 'oracle'
           elif i< sample2label_size:
               oracle_perfest = 'perf-est'
           else:
               oracle_perfest = 'candidate'
           if i < oracle_size-1:
               background_color.append('white')
           elif i < sample2label_size-1:
               background_color.append('lightgreen')
           else:
               background_color.append('yellow')
        else:
           oracle_perfest=''
           background_color.append('white')
        oracle_perfest_list.append(oracle_perfest)
        i = i + 1
        try:
           res= es.search(index=database_index,body={'query':{'match':{'filename':str(idx)}}},size=1)
        except:
           res = []
        if len(res) == 0 or res['hits']['total'] == 0:
            try:
                md = metadata_df[metadata_df['id'] == str(idx)]
                fname=str(idx)
                col_ids=metadata2show	
                for j in range(0, len(col_ids)-1):
                  if col_ids[j]=='filename':
                     metadata_row.append(str(idx))
                  else:
                     try:
                        p=(md[col_ids[j]]).values[0]
                        if (col_ids[j]=='ra' or col_ids[j]=='dec') and str(p).find(":") == -1:
                           metadata_row.append(round(float(p),5))
                        elif col_ids[j]=='prediction':
                           metadata_row.append(prediction)
                        elif col_ids[j]=='entropy':
                           metadata_row.append(entropy)
                        elif col_ids[j]=='label':
                           metadata_row.append(label)
                        elif col_ids[j]=='iteration':
                           metadata_row.append(iteration_num)
                        elif col_ids[j]=='set':
                           metadata_row.append(oracle_perfest)
                        else:
                           metadata_row.append(p)
                        #print(p)
                     except:
                        metadata_row.append('')
                if i <= n_display_limit:
                   metadata.append(metadata_row)
                   fname_list.append(str(idx))
                   label_list.append(label);
                   try:
                      comment_list.append(hit['_source']['comment'])
                   except:
                      comment_list.append('')
            except Exception as e:
                print (e)
        for hit in res['hits']['hits']:
            col_ids=metadata2show;
            for j in range(0, len(col_ids)):
                p=[]
                try:
                    p = hit['_source'][col_ids[j]]
                except Exception as e:
                    k=1
                if (col_ids[j]=='ra' or col_ids[j]=='dec') and str(p).find(":") == -1:
                    metadata_row.append(round(float(p),5))
                elif col_ids[j]=='prediction':
                    metadata_row.append(prediction)
                elif col_ids[j]=='entropy':
                           metadata_row.append(entropy)
                elif col_ids[j]=='label':
                    metadata_row.append(label)
                elif col_ids[j]=='iteration':
                    metadata_row.append(iteration_num)
                elif col_ids[j]=='set':
                    metadata_row.append(oracle_perfest)
                else:
                    metadata_row.append(p)
            if i <= n_display_limit:
               metadata.append(metadata_row)
               fname_list.append(str(idx))
               label_list.append(label);
               try:
                  comment_list.append(hit['_source']['comment'])
               except:
                  comment_list.append('')
    metadata_csv=folder_prefix+folder+learning_session_name + '/metadata_' + str(iteration_num) + '.csv'
    with open(metadata_csv,"w+") as md_csv:
       csvWriter = csv.writer(md_csv,delimiter=',')
       csvWriter.writerow(metadata2show)
       csvWriter.writerows(metadata)
    wavelengths = spectra.columns.values.tolist()
    wavelengths_str = json.dumps(wavelengths)
    html_code = html_template.substitute({"list": "".join(spectra_list),"folder":folder,"session":learning_session_name,"itnum":iteration_num,
"labels2add_fname": labels2add_fname,"spectra2add_fname":spectra2add_fname,"md": metadata,"comments": comment_list,"wavelengths": wavelengths_str,"fname": fname_list,"statistics":statistics,"prediction": prediction_list,"cat": cat_list,"lab": label_list,"oracle_perf": oracle_perfest_list,"random_sample_size": random_sample_size,"oracle_size": oracle_size,"mdcols": metadata2show,"raw_spectra_source":raw_spectra_source})
    try:
        spectra.to_csv("spectra.txt", header=False, index=False, sep=",")
    except Exception as e:
        print(e)
    return html_code

def number_of_lines(file):
    numlines = 0
    try:
       f = open(file,"r")
       b = f.readline()
       while b:
          b = f.readline()
          numlines = numlines+1
       f.close()
    except Exception as e:
        print(e)
    return numlines

def write_statistics(iteration_num,cat_list,statistics_csv,statistics,perf_est_from_previous_iteration):
    numlines = number_of_lines(statistics_csv)
    f= open(statistics_csv,"a+")
    if numlines==0:
        f.write('iteration,')
        for cat in cat_list:
            f.write(cat + ',')
        f.write('performance\n')
    if numlines==1:
       f.write(str(iteration_num)+',')
       for stat in statistics:
          f.write(str(stat)+',')
    if numlines>1:
       f.write(str(perf_est_from_previous_iteration)+'\n')
       f.write(str(iteration_num)+',')
       for stat in statistics:
          f.write(str(stat)+',')
    try:
       statistics_df = pd.read_csv(statistics_csv)
    except:
       statistics_df=[]
    return statistics_df

def plot_statistics(statistics_all):
    ax = plt.axes(xlabel='iteration', ylabel='estimated accuracy')
    ax.plot(statistics_all['iteration'].head(-1),statistics_all['performance'].head(-1),'x')
    plt.savefig('performance.pdf')

def save_to_database(labels2add_csv,cat_list,previous_iteration_spectra_file,learning_session_name,database_index):
    es=Elasticsearch([{'host':'localhost','port':9200}])
    f1=open(labels2add_csv,'r')
    b=f1.readline()
    f2=open(previous_iteration_spectra_file,'r')
    c=f2.readline()
    c=f2.readline()
    if (len(c))<2:
       c=f2.readline()
    while len(b)>1 and len(c)>1:
        b_array = b.split(',')
        c_array = c.split(',')
        fname = c_array[0]
        if len(b_array) > 1:
            if b_array[1]=="undefined":
                comment=""
            else:
                comment = b_array[1]
        else:
            comment = ""
        if len(b_array) > 0:
            try:
               label_num = int(b_array[0])
               label = cat_list[label_num]
               res= es.search(index=database_index,body={'query':{'match':{'filename':fname}}},size=1)
               if res['hits']['total'] > 0:
                  for hit in res['hits']['hits']:
                     filename= hit['_source']['filename']
                     if filename==fname:
                        idb = hit['_id']
                        es.update(index=database_index,doc_type='doc',id=idb,body={"doc": {"learning_session": learning_session_name,"label": label, "comment": comment,"label_"+learning_session_name: label, "comment_"+learning_session_name: comment}})
            except:
               print(fname+" not labelled.")
        b=f1.readline()
        c=f2.readline()
        if len(c)<2:
           c=f2.readline()
    f1.close()
    f2.close()

def label_random_samples(json_dict):
    poolnames=""
    batch_size=0
    if 'batch_size' in json_dict:
        batch_size = int(json_dict['batch_size'])
    if 'random_sample_size' in json_dict:
        random_sample_size = int(json_dict['random_sample_size'])
        batch_size = batch_size + random_sample_size
    if batch_size==0:
        batch_size=100
    if 'pool_csv' in json_dict:
        pool_csv = json_dict['pool_csv']
    elif 'csv_spectra_file' in json_dict:
        pool_csv = json_dict['csv_spectra_file']
    if 'poolnames_csv' in json_dict:
        poolnames_csv = json_dict['poolnames_csv']
        poolnames = pd.read_csv(poolnames_csv, index_col='id')
    else:
        poolnames_csv = pool_csv
        poolnames = pd.read_csv(pool_csv, index_col='id')
    ids = poolnames.index.values
    rnd_idx = np.random.choice(ids, size=batch_size)
    f= open("rand.csv","w+")
    for i in rnd_idx:
       f.write(i+'\n')
    f.close() 
    rand_csv = 'rand.csv'
    return rand_csv

def run_active_learning(json_config_file):
    json_dict = None
    with open(json_config_file, 'r') as f:
        json_dict = json.load(f)
    folder_prefix = '/data/vocloud/filesystem/DATA/'
    classes = []
    statistics = []
    statistics_all = []
    pe=""
    normalize = do_binning = remove_duplicates = False
    if 'learning_session_name' in json_dict:
        learning_session_name = json_dict['learning_session_name']
    else:
        learning_session_name = 'single_double_peak'
        print('No learning_session_name defined, switching to default: single_double_peak')  
    if 'folder' in json_dict:
        folder = json_dict['folder']
    else:
        folder = 'active-learning/'
    if 'database_index' in json_dict:
        database_index = json_dict['database_index']
    else:
        database_index = 'lamost-dr5-v3' 
    if 'iteration_num' in json_dict:
        iteration_num = int(json_dict['iteration_num'])
    else:
        iteration_num = 0
    if 'categories' in json_dict:
        cat_list = json_dict['categories']
    elif 'classes' in json_dict:
        cat_list = json_dict['classes']
    else:
        cat_list = ["other","single peak","double peak"]
    if 'candidate_classes' in json_dict:
        candidate_classes = json_dict['candidate_classes']
    else:
        candidate_classes = cat_list[-1]
    if 'classes' in json_dict:
         classes=json_dict['classes']
    if 'random_sample_size' in json_dict:
        random_sample_size = int(json_dict['random_sample_size'])
    else:
        random_sample_size = 3
	if random_sample_size < 3:
		random_sample_size = 3
		print('Random sample size adjusted to minimum = 3.')
    if 'batch_size' in json_dict:
        batch_size = int(json_dict['batch_size'])
    else:
        batch_size = 10
	if batch_size < 10:
		batch_size = 10
		print('Batch size adjusted to minimum = 10.')	
    if 'normalize' in json_dict:
        normalize = json_dict['normalize']
    if 'binning' in json_dict:
        do_binning = json_dict['binning']
    if 'remove_duplicates' in json_dict:
        remove_duplicates = json_dict['remove_duplicates']

    spectra2add_fname = 'spectra2add_' + learning_session_name + '_' + str(iteration_num) + '.csv' 
    labels2add_fname = 'labels_' + learning_session_name + '_' + str(iteration_num) + '.csv'
    labels2add = folder_prefix + folder + learning_session_name + '/labels_' + str(iteration_num-1) + '.csv'
    previous_iteration_spectra_file2 = folder_prefix + folder + learning_session_name + '/spectra2add_' + str(iteration_num-1)+'.csv'
    if 'training_set_csv' in json_dict:
           training_set_csv = json_dict['training_set_csv']
    else:
           training_set_csv = folder_prefix + folder + 'training-set-'+learning_session_name+'.csv'
    if not(os.path.exists(training_set_csv)):
        os.system("touch '{0}'".format(training_set_csv))
        cmd = "head -n1 '{0}' > '{1}'".format(folder_prefix + '/active-learning/training-set.csv',training_set_csv)
        os.system(cmd)
    if iteration_num > -1 or 'training_set_csv' in json_dict:
           try:
                  new_folder = folder_prefix + folder + learning_session_name
                  if not(os.path.exists(new_folder)):
                     cmd = "mkdir '{0}'".format(new_folder)
                     os.system(cmd)
                  training_set_csv1 = new_folder + '/training-set_0.csv'
           except:
                  training_set_csv1=training_set_csv[0:len(training_set_csv)-4]+'_0.csv'
           if not(os.path.exists(training_set_csv1)):
                  cmd = "cp '{0}' '{1}'".format(training_set_csv,training_set_csv1)
                  os.system(cmd)
    if iteration_num > 0:
       training_set_csv_old = training_set_csv1[0:len(training_set_csv1)-5]+str(iteration_num-1)+'.csv'
       training_set_csv_new = training_set_csv1[0:len(training_set_csv1)-5]+str(iteration_num)+'.csv'
       if not(os.path.exists(training_set_csv_new)):
           try:
              training_set_addition_csv = "training_set_addition.csv"
              pe=dh.prepare_spectra2add(previous_iteration_spectra_file2,labels2add,training_set_addition_csv,batch_size)
              cmd = "cp '{0}' '{1}' ".format(training_set_csv_old,training_set_csv_new)
              os.system(cmd)
              cmd = "tail -n+2 '{0}' >> '{1}'".format(training_set_addition_csv,training_set_csv_new)
              os.system(cmd)
              save_to_database(labels2add,cat_list,previous_iteration_spectra_file2,learning_session_name,database_index)
           except Exception as e:
              print (e)
              os.system("cp '{0}' '{1}' ".format(training_set_csv_old,training_set_csv_new))
       training_set_csv=training_set_csv_new
       metadata_csv_old=folder_prefix+folder+learning_session_name + '/metadata_' + str(iteration_num-1) + '.csv'
       try:
          md_old = pd.read_csv(metadata_csv_old,header=0)
          labels = pd.read_csv(labels2add,usecols=[0])
          labels.loc[-1] = ['label']
          md_old['label'] = labels
          md_old.to_csv(metadata_csv_old,header=True)
       except Exception as e:
                print (e)
    if 'show_candidates' in json_dict:
        show_candidates = json_dict['show_candidates']
    else:
        show_candidates = "no"
    if 'performance_estimation_csv' in json_dict:
        performance_estimation_csv = json_dict['performance_estimation_csv']
    else:
        performance_estimation_csv='perf-est.csv'
    if 'oracle_csv' in json_dict:
        oracle_csv = json_dict['oracle_csv']
    else:
        oracle_csv = "oracle.csv"
    to_label_csv = oracle_csv
    if 'labels2add_csv' in json_dict:
        labels2add_csv = json_dict['labels2add_csv']
        save_to_database(labels2add_csv,cat_list)
    if 'metadata2show' in json_dict:
        metadata2show = json_dict['metadata2show']
    else:
        metadata2show = ["filename","class","subclass","mag1","ra","dec","prediction","label","iteration","set"]
    if 'metadata2import' in json_dict:
        metadata2import = json_dict['metadata2import']
    else:
        metadata2import = ["id","obsid","designation","obsdate","lmjd","mjd","planid","spid","fiberid","ra_obs","dec_obs","snru","snrg","snrr","snri","snrz","objtype","class","subclass","z","z_err","magtype","mag1","mag2","mag3","mag4","mag5","mag6","mag7","tsource","fibertype","tfrom","tcomment","offsets","offset_v","ra","dec","fibermask"]
    poolnames_csv=""
    if 'poolnames_csv' in json_dict:
        poolnames_csv = json_dict['poolnames_csv']
    pool_csv = ""
    if 'pool_csv' in json_dict:
        pool_csv = json_dict['pool_csv']
        if poolnames_csv=="": poolnames_csv=pool_csv

    statistics_csv = folder_prefix + folder + learning_session_name + '/statistics.csv'
    if 'run_active_learning' in json_dict:
        run_active_learning = json_dict['run_active_learning']
        if run_active_learning=='y' or run_active_learning=='yes' or run_active_learning=='ano': 
           if training_set_csv == "" or iteration_num==0:
              oracle_csv=label_random_samples(json_dict)
              to_label_csv = oracle_csv
           else:
              statistics=activeCnn(pool_csv,training_set_csv,len(cat_list),cat_list,candidate_classes,batch_size,random_sample_size)
              statistics_all = write_statistics(iteration_num,cat_list,statistics_csv,statistics,pe)
              try:
                   plot_statistics(statistics_all)
              except:
                   print("Not enough data to plot performance.")
              to_label_csv = 'to_label_csv.csv'
              cmd = "cat '{0}' '{1}' > '{2}'".format(oracle_csv,performance_estimation_csv,to_label_csv)
              os.system(cmd)
    try:
        to_label_size = sum(1 for line in open(to_label_csv))
        oracle_size = sum(1 for line in open(oracle_csv))
    except:
        to_label_size = 0
        oracle_size = 0
    if 'batch_size' in json_dict:
        batch_size = int(json_dict['batch_size'])
        to_label_size2 = batch_size - random_sample_size
        if to_label_size!= to_label_size2:
           if to_label_size==0:
              to_label_csv=label_random_samples(json_dict)
              to_label_size = to_label_size2
    else:
        batch_size = random_sample_size + to_label_size
    csv_spectra_file2 = folder_prefix + folder + learning_session_name + '/spectra2add_' + str(iteration_num)+'.csv'
    b_read_spectra_from_fits_or_vot_files = 0
    if 'raw_spectra_source' in json_dict:
        raw_spectra_source = json_dict['raw_spectra_source']
    else:
        raw_spectra_source = 1
    if 'pool_csv' in json_dict:
        processed_df = dh.load_set_al(poolnames_csv,pool_csv,to_label_csv,csv_spectra_file2,raw_spectra_source)
        processed_df.columns = pd.to_numeric(processed_df.columns)
    else:
        spectra_list = dh.load_spectra_from_fits('.')
        b_read_spectra_from_fits_or_vot_files = 1
        processed_df = dh.to_dataframe(dh.process_set(spectra_list,
                                                  normalize=normalize,
                                                  binning=do_binning,
                                                  remove_duplicates=remove_duplicates),
                                   class_dict=classes)
    metadata_df=[]
    if 'csv_metadata_file' in json_dict:
        csv_metadata_file = json_dict['csv_metadata_file']
        metadata_df = dh.load_metadata_set(csv_metadata_file,metadata2import)
    else: 
        csv_metadata_file = 'metadata.csv'
        if b_read_spectra_from_fits_or_vot_files == 1:
            if 'metadata2import' in json_dict:
                metadata_df = dh.load_metadata_set(csv_metadata_file,metadata2import)
            else:
                metadata_df = dh.load_metadata_set0(csv_metadata_file)
    if show_candidates == 'yes':
            candidates_df = dh.load_set_al(poolnames_csv,pool_csv,"candidates.csv","csv_candidates_spectra.csv",raw_spectra_source)
            candidates_df.columns = pd.to_numeric(candidates_df.columns)
            processed_spectra2display = [processed_df,candidates_df]
            processed_df=pd.concat(processed_spectra2display)
    else:
        candidates_df = []
    if 'out_file' in json_dict:
        processed_df.to_csv("./" + json_dict['out_file'], header=False, index=True, index_label='id')
        to_votable(processed_df, 'meta.xml')
    if show_candidates == 'yes':
        cmd = "cat '{0}' '{1}' > '{2}'".format(to_label_csv,"candidates.csv","predictions.csv")
        os.system(cmd)
        prediction_file="predictions.csv"
    else:
        prediction_file=to_label_csv
    html_code = _generate_spectra(folder_prefix,folder,processed_df,database_index,metadata_df,classes,prediction_file,random_sample_size,oracle_size,cat_list,metadata2show,iteration_num,spectra2add_fname,labels2add_fname,raw_spectra_source,statistics,learning_session_name)
    with open("./index.html", "w") as file:
        file.write(html_code)

if __name__ == '__main__':
    main()
