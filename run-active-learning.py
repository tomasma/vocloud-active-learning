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

__LINK_TEMPLATE = string.Template('<option selected id="${spectrum_name}_link" style="background-color:${background_color};">${spectrum_name}</option>\n')
__script_dir = os.path.dirname(os.path.realpath(__file__))


def main():
    if len(sys.argv) < 2:
        raise ValueError('Must pass at least one input file')
    for file in sys.argv[1:]:
        run_preprocessing(file)


def _generate_spectra(spectra,database_index,metadata_df,classes,prediction_file,random_sample_size,oracle_size,cat_list,metadata2show,iteration_num,spectra2add_fname,labels2add_fname,raw_spectra_source,statistics):
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
                        print(p)
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
    categories = spectra.columns.values.tolist()
    if categories[-1] == 'class':
        categories_str = json.dumps(categories[:-1])

        html_code = html_template.substitute(
            {"list": "".join(spectra_list), "comments": comment,"cats": categories_str,"fname": fname,"statistics":statistics,"prediction": prediction,"oracle_perfest": oracle_perfest,"raw_spectra_source":raw_spectra_source})
    else:
        categories_str = json.dumps(categories)
        html_code = html_template.substitute({"list": "".join(spectra_list), "labels2add_fname": labels2add_fname,"spectra2add_fname":spectra2add_fname,"md": metadata,"comments": comment_list,"cats": categories_str,"fname": fname_list,"statistics":statistics,"prediction": prediction_list,"cat": cat_list,"lab": label_list,"oracle_perf": oracle_perfest_list,"random_sample_size": random_sample_size,"oracle_size": oracle_size,"mdcols": metadata2show,"raw_spectra_source":raw_spectra_source})
    try:
        spectra.to_csv("spectra.txt", header=False, index=False, sep=",")
    except Exception as e:
        print(e)
    return html_code

def save_to_database(labels2add_csv,cat_list):
    es=Elasticsearch([{'host':'localhost','port':9200}])
    f=open(labels2add_csv,'r')
    b=f.readline()
    while len(b)>1:
        b_array = b.split(',')
        fname = b_array[0]
        if len(b_array) > 2:
            if b_array[2]=="undefined":
                comment=""
            else:
                comment = b_array[2]
        else:
            comment = ""
        if len(b_array) > 1:
            label_num = int(b_array[1])
        else:
            label_num = len(cat_list)-1
        label = cat_list[label_num]
        res= es.search(index='lamost-dr5-v3',body={'query':{'match':{'filename':fname}}},size=1)
        if res['hits']['total'] > 0:
            for hit in res['hits']['hits']:
                filename= hit['_source']['filename']
                if filename==fname:
                     idb = hit['_id']
                     es.update(index='lamost-dr5-v3',doc_type='doc',id=idb,body={"doc": {"label": label, "comment": comment }})
        b=f.readline()
    f.close()

def to_votable(data, file_name):
    votable = VOTableFile()
    resource = Resource()
    votable.resources.append(resource)
    table = Table(votable)
    resource.tables.append(table)
    columns = data.columns
    if data.columns[-1] == 'class':
        columns = columns[:-1]
    fields = [Field(votable, name="placeholder", datatype="char", arraysize='*'),
        Field(votable, name="intensities", datatype="double", arraysize='*')]
    table.fields.extend(fields)
    table.create_arrays(1)
    table.array[0] = ("placeholder", columns.tolist())
    votable.to_xml(file_name)

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
    elif 'csv_spectra_file_names' in json_dict:
        poolnames_csv = json_dict['csv_spectra_file_names']
        poolnames = pd.read_csv(poolnames_csv, index_col='id')
    else:
        poolnames_csv = pool_csv
        csv_spectra_file_names = pool_csv
        poolnames = pd.read_csv(pool_csv, index_col='id')
    ids = poolnames.index.values
    rnd_idx = np.random.choice(ids, size=batch_size)
    f= open("rand.csv","w+")
    for i in rnd_idx:
       f.write(i+'\n')
    f.close() 
    rand_csv = 'rand.csv'
    return rand_csv

def run_preprocessing(input_file):
    json_dict = None
    normalize = do_binning = remove_duplicates = False
    with open(input_file, 'r') as f:
        json_dict = json.load(f)
    if 'normalize' in json_dict:
        normalize = json_dict['normalize']
    if 'binning' in json_dict:
        do_binning = json_dict['binning']
    if 'remove_duplicates' in json_dict:
        remove_duplicates = json_dict['remove_duplicates']
    classes = []
    statistics = []
    if 'classes' in json_dict:
         classes=json_dict['classes']

    performance_estimation_csv='perf-est.csv'
    oracle_csv=[]
    run_active_learning = []
    training_set_csv = ""
    learning_session_name = 'single_double_peak'
    if 'database_index' in json_dict:
        database_index = json_dict['database_index']
    else:
        database_index = 'lamost-dr5-v3'
    if 'learning_session_name' in json_dict:
        learning_session_name = json_dict['learning_session_name']
    if 'iteration_num' in json_dict:
        iteration_num = int(json_dict['iteration_num'])
    else:
        iteration_num = 0
    spectra2add_fname = 'spectra2add_' + learning_session_name + '_' + str(iteration_num) + '.csv' 
    if 'spectra2add_filename' in json_dict:
        spectra2add_fname = spectra2add_filename
    labels2add_fname = 'labels_' + learning_session_name + '_' + str(iteration_num) + '.csv'
    if 'labels2add_filename' in json_dict:
        labels2add_fname = json_dict['labels2add_filename']
    if 'training_set_csv' in json_dict:
           training_set_csv = json_dict['training_set_csv']
    if 'training_set_addition_csv' in json_dict:
        training_set_addition_csv = json_dict['training_set_addition_csv']
        if 'training_set_csv' in json_dict:
           training_set_csv = json_dict['training_set_csv']
           training_set_csv_orig = training_set_csv[0:len(training_set_csv)-4]+'_orig.csv'
           cmd = "cp '{0}' '{1}' ".format(training_set_csv,training_set_csv_orig)
           os.system(cmd)
           cmd = "tail -n+2 '{0}' >> '{1}'".format(training_set_addition_csv,training_set_csv)
           os.system(cmd)
        else:
           json_dict['training_set_csv']=training_set_addition_csv
    if 'candidates_csv' in json_dict:
        candidates_csv = json_dict['candidates_csv']
    if 'show_candidates' in json_dict:
        show_candidates = json_dict['show_candidates']
    else:
        show_candidates = "no"
    if 'performance_estimation_csv' in json_dict:
        performance_estimation_csv = json_dict['performance_estimation_csv']
    if 'oracle_csv' in json_dict:
        oracle_csv = json_dict['oracle_csv']
    else:
        oracle_csv = "oracle.csv"
    to_label_csv = oracle_csv
    if 'categories' in json_dict:
        cat_list = json_dict['categories']
    else:
        cat_list = ["other","single peak","double peak"]
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
    if 'run_active_learning' in json_dict:
        run_active_learning = json_dict['run_active_learning']
        if run_active_learning=='y' or run_active_learning=='yes' or run_active_learning=='ano': 
           if training_set_csv == "":
              oracle_csv=label_random_samples(json_dict)
              to_label_csv = oracle_csv
           else:
              statistics=activeCnn()
              to_label_csv = 'to_label_csv.csv'
              cmd = "cat '{0}' '{1}' > '{2}'".format(oracle_csv,performance_estimation_csv,to_label_csv)
              os.system(cmd)
    if 'random_sample_size' in json_dict:
        random_sample_size = int(json_dict['random_sample_size'])
    else: 
           random_sample_size=0
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
    csv_spectra_file_names=""
    if 'csv_spectra_file_names' in json_dict:
        csv_spectra_file_names = json_dict['csv_spectra_file_names']
    elif 'poolnames_csv' in json_dict:
        csv_spectra_file_names = json_dict['poolnames_csv']
    if 'csv_spectra_file2' in json_dict:
        csv_spectra_file2 = json_dict['csv_spectra_file2']
    else:
        csv_spectra_file2 = 'csv_spectra_file2.csv'
    b_read_spectra_from_fits_or_vot_files = 0
    if 'raw_spectra_source' in json_dict:
        raw_spectra_source = json_dict['raw_spectra_source']
    else:
        raw_spectra_source = 1
    if 'csv_spectra_file' in json_dict:
        csv_spectra_file = json_dict['csv_spectra_file']
        if csv_spectra_file_names=="": csv_spectra_file_names=csv_spectra_file
        processed_df = dh.load_set3(csv_spectra_file_names,csv_spectra_file,to_label_csv,csv_spectra_file2,raw_spectra_source)
        processed_df.columns = pd.to_numeric(processed_df.columns)
    elif 'pool_csv' in json_dict:
        csv_spectra_file = json_dict['pool_csv']
        if csv_spectra_file_names=="": csv_spectra_file_names=csv_spectra_file
        processed_df = dh.load_set3(csv_spectra_file_names,csv_spectra_file,to_label_csv,csv_spectra_file2,raw_spectra_source)
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
    with_output=0
    if 'show_candidates' in json_dict:
        show_candidates = json_dict['show_candidates']
        if show_candidates == 'yes':
            candidates_df = dh.load_set3(csv_spectra_file_names,csv_spectra_file,"candidates.csv","csv_candidates_spectra.csv",raw_spectra_source)
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
		
    #next iteration json configuration
    with open('new_config.json', 'w') as f2:
        x = json_dict
        if "iteration_num" in x:
           x["iteration_num"] = x["iteration_num"] + 1
        else:
           x["iteration_num"] = 1
        x["training_set_addition_csv"] = spectra2add_fname
        x["spectra2add_fname"] = 'spectra2add_' + learning_session_name + '_' + str(iteration_num+1) + '.csv'
        x["labels2add_fname"] = 'labels_' + learning_session_name + '_' + str(iteration_num+1) + '.csv'
        json.dump(x,f2,indent=4)
    if 'import2elastic' in json_dict:
        csv2elastic(spectra)
    else:
        header = processed_df.columns
        html_code = _generate_spectra(processed_df,database_index,metadata_df,classes,prediction_file,random_sample_size,oracle_size,cat_list,metadata2show,iteration_num,spectra2add_fname,labels2add_fname,raw_spectra_source,statistics)
        with open("./index.html", "w") as file:
            file.write(html_code)

if __name__ == '__main__':
    main()
