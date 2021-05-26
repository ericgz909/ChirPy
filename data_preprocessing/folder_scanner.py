# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:12:12 2021

@author: Slawa
"""

import os
import sys
Path=os.path.dirname((os.path.abspath(__file__)))
sys.path.append(Path)

from Bird_sound_class import sound_folder

"""sound_folder() is taking either a folder argument sound_folder(folder=folder), in that case
the entire folder is scanned,
or a list of files corresponding to the same bird sound_folder(fileslist=[file1,file2,...])), 
in that case only these files are used in analysis"""

files=[r'F:\bootcamp data\train_short_audio\aldfly\small\XC2628.ogg',
       r'F:\bootcamp data\train_short_audio\aldfly\small\XC16964.ogg']

BSF=sound_folder()
#Number_of_subfolders=... to limit the number of subfolders to scan through
[individuals,average]=BSF.run(outputfilename='features')
#individuals are pandas data frame for single files
#average is the pandas data frame for the averaged spectra across all files
# you can specify the output file name BSF.run(outputfilename='file name')