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
or a list of files corresponding to the same bird (ound_folder(files=[file1,file2,...])), 
in that case only these files are used in analysis"""

BSF=sound_folder()
[individuals,average]=BSF.run()
#individuals are pandas data frame for single files
#average is the pandas data frame for the averaged spectra across all files