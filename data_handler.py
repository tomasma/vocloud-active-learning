import os
import io
import random
import itertools
import astropy.io.fits as pyfits
import pandas as pd
import pprint
import linecache

import sklearn.preprocessing as preprocessing
import numpy as np
from astropy.io import fits
from elasticsearch import Elasticsearch

from spectraml import analyzer2
from spectraml import ondrejov
from spectraml import lamost

def load_spectra_from_fits(uri):
    #return _to_array(_parse_all_fits(uri))
    return _parse_all_fits(uri)

def process_set(spectra, normalize=True, binning=True, remove_duplicates=True, delimiter=','):
    if binning:
        spectra = __spectra_rebinning(spectra)
    if normalize:
        _normalize(spectra)
    # csv_name = 'processed.csv'
    #_write_fits_csv(spectra, csv_name)
    return spectra

def load_set_al(csv_spectra_file_names,csv_spectra_file,oracle_csv,csv_spectra_file2,raw_spectra_source,format='csv', header=True, delimiter=','):
    n_display_limit=1000
    csfn = pd.read_csv(csv_spectra_file_names,memory_map=True)
    f=open(oracle_csv,'r')
    b=f.readline()
    cmd = "head -n1 '{0}' > '{1}'".format(csv_spectra_file,csv_spectra_file2)
    os.system(cmd)
    f2=open(csv_spectra_file2,"a+")
    f3=open('raw_spectra_wavelengths.txt',"a+")
    f4=open('raw_spectra_data.txt',"a+")
    orig_fits=[]
    i=0
    while b and i<n_display_limit:
        b=b.rstrip('\n')
        b_array=b.split(',')
        b=b_array[0]

        if raw_spectra_source==1:
           name_array=b.split('-')
           name_array2=name_array[2]

           folder0=name_array[2]
           folder=folder0[:-5]
           fits_file='/i/antares/public/LAMOST-DR2/fits/'+folder+'/'+b+'.fits'
           fits_file_zipped=fits_file+'.gz'
           try:
              parsed_fits=_parse_fits1(fits_file)
              parsed_fits['header'].tofile(f3, sep=",", format="%s")
              print('',file=f3)
              parsed_fits['data'].tofile(f4, sep=",", format="%s")
              print('',file=f4)
           except Exception as e:
                print (e)
        idx=csfn.index[csfn['id'] == b].tolist()
        if len(idx)>0:
            f2.write(linecache.getline(csv_spectra_file,idx[0]+2))
        b=f.readline()
        i=i+1
    f.close()
    f2.close()
    f3.close()
    f4.close()
    return pd.read_csv(csv_spectra_file2, header=0 if header else None,
                       sep=None, dtype=None, na_values='?',
                       skipinitialspace=True, index_col='id',engine='python')

def prepare_spectra2add(csv_spectra_file2add,labels2add,training_set_addition_csv,oracle_size):
    pe=""
    f1=open(csv_spectra_file2add,'r')
    f2=open(labels2add,'r')
    f3=open(training_set_addition_csv,'w+')
    b=f1.readline()
    b_array=b.split(',',1)
    f3.write("id,label,"+b_array[1])
    b=f1.readline()
    i=0
    while (b or c) and i<oracle_size:
           c=f2.readline()
           c_array=c.split(',')
           c1=c_array[0]
           if len(c_array)>1:
              if len(c1)>0:
                  b_array=b.split(',',1)
                  b1=b_array[0]
                  b2=b_array[1]
                  f3.write(b1+","+c1+","+b2)
           elif len(c1)>0:
                  pe = c1
           b=f1.readline()
           i=i+1
    f1.close()
    f2.close()
    f3.close()
    return float(pe)/100

def load_metadata_set0(uri, format='csv', header=True, delimiter=','):
    md = pd.read_csv(uri, usecols=['id','ra','dec'],header=0 if header else None,
                       sep=None, dtype=None, na_values='?',
                       skipinitialspace=True, index_col=False)
    for i in range(len(md)):
        x=md.iloc[i,0]
        y0=x.rsplit("/",1)[-1]
        y1=y0.split(".",1)[0]
        md.iloc[i,0]=y1
    return md

def load_metadata_set(uri,metadata2import):
    return pd.read_csv(uri, usecols=metadata2import,index_col=None)

def to_dataframe(spectra_list, class_dict=None):
    indices = [spectrum['id'] for spectrum in spectra_list]
    columns = spectra_list[0]['header']
    data = [spectrum['data'] for spectrum in spectra_list]
    spectra_df = pd.DataFrame(data=data, columns=columns, index=indices)
    if len(class_dict) > 0:
        classes = [class_dict[index] for index in indices]
        spectra_df.insert(len(spectra_df.columns), 'class', classes)
    return spectra_df

def _to_array(fits_list):
    for fits in fits_list:
        data = []
        header = []
        for length, intensity in fits['data']:
            data.append(intensity)
            header.append(length)
        fits['data'] = data
        fits['header'] = header
    return fits_list

def _to_array1(fits):
    data = []
    header = []
    for length, intensity in fits['data']:
        data.append(intensity)
        header.append(length)
    fits['data'] = data
    fits['header'] = header
    return fits

def __spectra_rebinning(fits_list):
    '''Bin the incoming data (expecting two columns [wavelength, intensity]) based on the difference
    between two subsequent points. If the difference between avg of current bin and current point
    exceeds 0.25 we will start new bin

    Note that fits['header'] is array of wavelengths and fits['data'] is array of intensities which
    conforms to the wavelengths.
    '''
    result = []
    firsts = [x['header'][0] for x in fits_list] # if x['header'][-1] <= 6500]
    lasts = [x['header'][-1] for x in fits_list] # if x['header'][-1] >= 6700]
    start = max(firsts)
    stop = min(lasts)
    binned_header = np.linspace(start, stop, 1980)
    for fits in fits_list:
        fits_data = fits['data']
        fits_header = fits['header']
        indexes = [idx for idx,x in enumerate(fits_header) if start <= x <= stop]
        fits_header = [fits_header[idx] for idx in indexes]
        fits_data = [fits_data[idx] for idx in indexes]
        binned_data = np.interp(binned_header, xp=fits_header, fp=fits_data)
        binned_dictionary = {'data': binned_data, 'id': fits['id'], 'header': binned_header}
        result.append(binned_dictionary)
    return result


def _normalize(fits_list, norm='l2'):
    '''normalize data'''
    # min_max_scaler = preprocessing.MinMaxScaler()
    data_list = [fits['data'] for fits in fits_list]
    preprocessed_data = preprocessing.normalize(data_list, norm=norm)
    for idx, item in enumerate(preprocessed_data):
        fits_list[idx]['data'] = item


def _parse_all_fits(uri):
    parsed_fits = []
    #if 'folder' in json_dict:
    es=Elasticsearch([{'host':'localhost','port':9200}])
    metadata = open('metadata.csv', 'a+')
    print('id,ra,dec',file=metadata)
    for root, dirs, files in os.walk(uri):
        fits_files = [file for file in files if file.endswith('.fits')]
        if len(fits_files) == 0: continue
        for fits_file in fits_files:
            try:
                fits = {}
                fits["data"] = _parse_fits(os.path.join(root, fits_file))
                fits["id"] = os.path.splitext(fits_file)[0]
                _parse_fits_metadata(os.path.join(root, fits_file),metadata)
                parsed_fits.append(_to_array1(fits))
            except:
                try:
                    identifier, wave, flux = lamost.read_spectrum(os.path.join(root, fits_file))
                except:
                    identifier, wave, flux = ondrejov.read_spectrum(os.path.join(root, fits_file))
                try:
                    fits["id"] = os.path.splitext(fits_file)[0]
                    fits["data"] = flux
                    fits["header"] = wave
                    _parse_fits_metadata(os.path.join(root, fits_file),metadata)
                    parsed_fits.append(fits)
                except Exception as e: print(str(e) + "for :" + str(fits_file))

    return parsed_fits


def _parse_fits(uri):
    fits = pyfits.open(uri, memmap=False)
    if len(fits) > 1:
        dat = fits[1].data
    else:
        dat=[]
        dat = fits[0].data[0]
    fits.close()
    return dat.tolist()

def _parse_fits1(uri):
    fits = {}
    try:
        identifier, wave, flux = lamost.read_spectrum(uri)
    except:
        identifier, wave, flux = ondrejov.read_spectrum(uri)
    try:
        fits["id"] = identifier
        fits["data"] = flux
        fits["header"] = wave
        return fits
    except Exception as e: 
        fits["id"] = []
        fits["data"] = []
        fits["header"] = []
    return fits

def _parse_fits_metadata(uri,metadata):
    fits = pyfits.open(uri, memmap=False)
    if len(fits)>1:
        header = fits[1].header
        print(header['TITLE'],header['RA'],header['DEC'], sep=',', file= metadata)
    else:
        header = fits[0].header
        print(header['FILENAME'],header['RA'],header['DEC'], sep=',', file= metadata)
    fits.close()
    return header
