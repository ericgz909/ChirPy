# -*- coding: utf-8 -*-
"""
Created on Mon May 24 12:12:22 2021

@author: Slawa
"""

import os
import sys
Path=os.path.dirname((os.path.abspath(__file__)))
sys.path.append(Path)

from Bird_sound_class import sound_folder


BSF=sound_folder()
BSF.run()