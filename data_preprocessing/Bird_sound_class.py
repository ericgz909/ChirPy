# -*- coding: utf-8 -*-
"""
class for sound files processing

@author: Slawa
"""

import soundfile as sf
import os
import sys
Path=os.path.dirname((os.path.abspath(__file__)))
sys.path.append(Path)
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
from scipy import signal, ndimage
from scipy.fftpack import fft, fftshift, ifft, ifftshift
from scipy.stats.mstats import gmean
import pandas as pd


class Bird_sound_processor():
    def __init__(self,file=None):
        """file is an audio file path. if None, then a window to chose file will pop up"""
        self.favdata=[]
        self.open(file)
        
    def open(self,file=None):
        """imports an audio file"""
        if file == None:
            file=filedialog.askopenfilename(title='choose sound file')
        self.file=file
        self.FilTdata=[]
        #check that the path is not empty
        if file == '':
            self.data=[]
            self.samplerate=0
            print('wrong file path')
            #!!!ideallly add raise error
        else:
            self.data, self.samplerate = sf.read(file)
            
                
    def FFT_step(self,tdata):
        """computes window fft for Tdata (the lengtht should be a power of 2 for optimum performance)"""
        fdata=fft(tdata,axis=1)
        return fdata
    
    def IFFT_step(self,wdata):
        """computes ifft for wdata"""
        tdata=ifft(wdata,axis=1)
        return tdata
    
    def split(self,Y,size):
        """splits an np.array to a list of arrays with size = size
        if the last array is smaller than size, it is ignored"""
        # comment: an overlap can be added in the splitting process
        N=int(np.floor(len(Y)/size))
        Y1=Y[:N*size]
        return Y1.reshape((N,-1))
    
    def FFT(self,size=2**15,smoothing=True,Gsigma=4):
        #,averaging='arithmetic'
        """splits data in parts with size length each and performs FFT for each of them
        averaging is either geometric or arithmetical
        FYI: size=2**15 is about 1 s at 32000 Hz sampling rate
        smoothing=True to apply gaussian filter to the spectral result
        Gsigma is the standard deviation for Gaussian kernel
        """
        Sdata=self.split(self.data,size)
        Fsdata=self.FFT_step(Sdata)
        AbsFdata=np.abs(Fsdata) #spectral amplitude
        arth_aver=np.mean(AbsFdata,axis=0)
        geom_aver=gmean(AbsFdata,axis=0)
        if np.sum(arth_aver) > np.sum(geom_aver):
            self.favdata=arth_aver
        else:
            self.favdata=geom_aver
            
        if smoothing:
            self.favdata=ndimage.gaussian_filter1d(self.favdata,Gsigma)
        
        Y0=self.favdata.copy()
        Y0/= np.abs(Y0).max()
        Y=Y0[:int(len(Y0)/2)]
        dt=1/self.samplerate #time step
        df=1/dt/(len(Y0)-1) #frequency step
        X=np.arange(len(Y))*df
        self.f=X
        self.If=Y
            
    def filter(self,intervals):
        """bandpass filtering leaving intervals
        intervals=[[f_low1,f_high1],[f_low2,f_high2],...]"""
        
            
    def exportFFT(self,file=None):
        """exports the averaged FFT
        in the form [frequency,data]"""
        X=self.f
        Y=self.If
        np.save(self.file[:-4]+'_averfft.npy',[X,Y])
        
    def loadFFT(self):
        """loads the saved FFT"""
        [X,Y]=np.load(self.file[:-4]+'_averfft.npy')
        self.f=X
        self.If=Y
        
        
    def exportFFTparam(self,Widthlevels=[0.5,np.exp(-2)],Fcut=100):
        """exports FFT parameters"""
        X0=self.f
        Y0=self.If
        ind=X0>Fcut
        X=X0[ind]
        Y=Y0[ind]
        Npeak=Y.argmax()
        Fm=X[Npeak] #peak frequency
        Width=np.zeros(len(Widthlevels))
        StrLevels=''
        for i in range(len(Widthlevels)):
            Nh=np.argwhere(Y[Npeak:]<Widthlevels[i])[0]
            Nl=np.argwhere(Y[:Npeak]<Widthlevels[i])[-1]
            Width[i]=X[Npeak+Nh]-X[Npeak+Nl]
            StrLevels+= ' '+str(Widthlevels[i])
        np.savetxt(self.file[:-4]+'_fftparams.dat', [Fm,Width.reshape((-1,1))],
                        header='peak frequency Hz '+ 
                        'width at levels' + StrLevels
                        ,delimiter='\t',comments='')
    
    def find_main_Tslice(self,Fband,size=2**15,cutlevel=0.06,maxTwindow=6,durationLevel=0.08):
        """find the most pronounced trmporal slice in the Fband=[Fmin,Fmax] frequency window
        finds only one most pronounced feature
        maxTwindow [seconds] is the maximum window to look for
        keep durationLevel > cutlevel"""
        #band pass filtering
        def Ffilter(F,fmin,fmax):
            """filter function"""
            return np.logical_and(F<fmax,F>fmin) #simples option (but should be enough for this purpose)
        Sdata=self.split(self.data,size)
        Fsdata=self.FFT_step(Sdata)
        dt=1/self.samplerate #time step
        df=1/dt/(size-1) #frequency step
        W=np.concatenate((np.arange(int(size/2))*df,np.arange(-int(size/2),0)*df))
        Filter=Ffilter(W,Fband[0],Fband[1])+Ffilter(W,-Fband[1],-Fband[0])
        FilFdata=Fsdata*Filter
        
        plt.figure(4)
        plt.clf()
        plt.yscale('log')
        for i in range(len(Fsdata)):
            plt.plot(np.abs(Fsdata[i]))
        plt.plot(np.abs(FilFdata[0]))
        
        #find a dominant temporal slice
        FilTdata=np.real(self.IFFT_step(FilFdata))
        self.FilTdata=FilTdata
        self.FilPower=np.abs(FilTdata)**2
        Gsigma=int(5*self.samplerate/1000)
        self.FilPowerAv=ndimage.gaussian_filter1d(self.FilPower,Gsigma)
        FS=self.FilPowerAv.reshape((1,-1))[0].copy() #filtered power
        Npeak=FS.argmax() #peak position
        Slice=FS[int(max((0,Npeak-maxTwindow/2/dt))):int(min((len(FS),Npeak+maxTwindow/2/dt)))]
        Slice/=Slice.max()
        ind0=np.argwhere(Slice>cutlevel)
        if len(ind0)>3:
            Slice=Slice[ind0[0][0]:ind0[-1][0]]
        self.bestslice=Slice
        
        plt.figure(5)
        plt.clf()
        t=np.arange(len(Slice))*dt
        plt.plot(t,Slice)
        
        if np.modf(np.log2(len(Slice)))[0] > 0:
            N2=2**int(np.ceil(np.log2(len(Slice)))) #new length =2**
            Slice1=np.concatenate((Slice,np.zeros(N2-len(Slice))))
        else:
            N2=len(Slice)
            Slice1=Slice
            
        #feature duration
        Nspeak=Slice1.argmax()
        if Nspeak==0:
            Nsmin=0
        else:
            if len(np.argwhere(Slice1[:Nspeak]<durationLevel)) == 0:
                Nsmin=0
            else:
                Nsmin=np.argwhere(Slice1[:Nspeak]<durationLevel)[-1][0]
        Nsmax=Nspeak+np.argwhere(Slice1[Nspeak:]<durationLevel)[0][0]
        self.duration_envelope=(Nsmax-Nsmin)*dt
        
        #find typical frequency in the slice
        SF=ndimage.gaussian_filter1d(np.abs(fft(Slice1)),3)
        SF/=SF.max()
        plt.figure(6)
        plt.clf()
        plt.plot(SF)

        
        SFm=np.roll(SF,-1)
        SFp=np.roll(SF,1)
        N12=int(len(SF)/2) # to cut a meaninful half of the spectrum
        ind0=np.logical_and(SF[:N12]>=SFm[:N12],SF[:N12]>=SFp[:N12])
        indm=np.argwhere(ind0==True)[1:,0]
        df2=1/dt/(N2-1) #frequency step
        if len(indm)<2:
            Nmax=0
            self.Fpeak_envelope=0
        else:
            Nmax=SF[indm].argmax()
            self.Fpeak_envelope=df2*(indm[Nmax]-1)
        
            
    def show_T(self,title=None):
        """plots the temporal signal"""
        if len(self.data) == 0:
            print('no data loaded')
        else:
            Y=self.data.copy()
            Y/= np.abs(Y).max()
            X=np.arange(len(Y))/self.samplerate
            ylable='amplitude (normalized)'
            xlable='time (s)'
            plt.plot(X,Y,linewidth=3)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel(xlable,fontsize=18)
            plt.ylabel(ylable,fontsize=18)
            if not title == None:
                plt.title(title,fontsize=18)
            if len(self.FilTdata)>0:
                Y1=self.FilTdata.reshape((1,-1))[0]
                Y1/= np.abs(Y1).max()
                X1=np.arange(len(Y1))/self.samplerate
                Y2=self.FilPower.reshape((1,-1))[0]
                Y2/= np.abs(Y2).max()
                Y3=self.FilPowerAv.reshape((1,-1))[0]
                Y3/= np.abs(Y3).max()
                plt.plot(X1,Y1,linewidth=3)
                plt.plot(X1,Y2,linewidth=3)
                plt.plot(X1,Y3,linewidth=3)
                plt.legend(('in','filtered','filtered power'))
                
    def show_W(self,title=None,logscale=True):
        """plots the spectral signal"""
        if len(self.favdata) == 0:
            print('no spectral data run .FFT first')
        else:
            Y0=self.favdata.copy()
            Y0/= np.abs(Y0).max()
            Y=Y0[:int(len(Y0)/2)]
            # DT=len(Y)/self.samplerate #time window
            dt=1/self.samplerate #time step
            df=1/dt/(len(Y0)-1) #frequency step
            X=np.arange(len(Y))*df
            ylable='amplitude (normalized)'
            xlable='frequency (Hz)'
            plt.plot(X,Y,linewidth=3)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel(xlable,fontsize=18)
            plt.ylabel(ylable,fontsize=18)
            if logscale:
                plt.yscale('log')
            if not title == None:
                plt.title(title,fontsize=18)
    
    def convert_export(self,Type='wav'):
        """mostly for my private use
        converts to another data format according to Type using soundfile (see it for supported types)"""
        file2=self.file.split('.')[0]+'.'+Type
        sf.write(file2,self.data,self.samplerate)
        
    def export_params(self,Widthlevels=[0.5,0.8],Fcut=100):
        """export parameters
        Fcut is the minimum frequency to take into account for FFT"""
        X0=self.f
        Y0=self.If
        ind=X0>Fcut
        X=X0[ind]
        Y=Y0[ind]
        Npeak=Y.argmax()
        Fm=X[Npeak] #peak frequency
        self.Fpeak=Fm
        Width=np.zeros(len(Widthlevels))
        Fmean=np.zeros(len(Widthlevels))
        for i in range(len(Widthlevels)):
            if len(np.argwhere(Y[Npeak:]<Widthlevels[i]))==0:
                Nh=len(Y[Npeak:])-1
            else:
                Nh=np.argwhere(Y[Npeak:]<Widthlevels[i])[0]
            if Npeak == 0:
                Nl=0
            else:
                if len(np.argwhere(Y[:Npeak]<Widthlevels[i]))==0:
                    Nl=0
                else:
                    Nl=np.argwhere(Y[:Npeak]<Widthlevels[i])[-1]
            Width[i]=X[Npeak+Nh]-X[Nl]
            Fmean[i]=(X[Npeak+Nh]+X[Nl])/2
            
        self.width=Width
        self.fmean=Fmean
        
        Titles=['peak frequency Hz','width at level ' + str(Widthlevels[0]),'width at level ' + str(Widthlevels[1]),
                'fmean at level ' + str(Widthlevels[0]),'fmean at level ' + str(Widthlevels[1]),
                'pulse duration','pulse rep rate']
        values=[Fm,Width[0],Width[1],
                Fmean[0],Fmean[1],
                self.duration_envelope,self.Fpeak_envelope]
        
        np.savetxt(self.file[:-4]+'_params.dat', np.array(values).reshape((1,-1)),
                        header='\t'.join(Titles),
                        delimiter='\t',comments='')
        return [['file']+Titles,[self.file.split('/')[-1][:-4]]+values]


#_____________________________________
    
class sound_folder():
    def __init__(self,folder=None,fileslist=[]):
        """folder is an audio folder path. if None, then a window to choose will pop up
        fileslist is the list of full files path for the same bird"""
        self.FFT=[]
        if fileslist==[]:
            self.open(folder)
            self.usefiles=False
        else:
            self.usefiles=True
            self.files=fileslist
        
    def open(self,folder=None,file_extention='ogg'):
        """opens and scans the folder for files"""
        if folder == None:
            folder=filedialog.askdirectory(title='folder with sound files')
        self.folder=folder
        files0=list(os.scandir(folder))
        files=[]
        for f in files0:
            if f.name.split('.')[-1] == file_extention:
                files.append(f.name)
        self.files=files
        
    def FFT_av(self,size=2**15,averaging='arithmetic',postaveraging='geometric'):
        """computes an averaged FFT for all files in the directory and saves it
        size is the data slicing parameter (size of each temporal slice)"""
        FFTs=[]
        for f in self.files:
            if self.usefiles:
                file=f
            else:
                file=self.folder+'/'+f
            BS=Bird_sound_processor(file)
            BS.FFT(size=size)
            BS.exportFFT()
            FFTs.append(BS.If)
        #computing the averaged spectrum of all of the files in the directory
        if averaging == 'geometric':
            FFTfull=gmean(FFTs,axis=0)
        else:
            FFTfull=np.mean(FFTs,axis=0)
        FFTfull/=FFTfull.max()
        if self.usefiles:
            np.save(file.split('.')[0]+'_fullaverfft.npy',[BS.f,FFTfull])
        else:
            np.save(self.folder+'/'+self.folder.split('/')[-1]+'_fullaverfft.npy',[BS.f,FFTfull])
        self.FFT=FFTfull
        self.f=BS.f
    
    def export_FFTparams(self,Widthlevels=[0.5,0.8],Fcut=100):
        """export FFT parameters
        Fcut is the minimum frequency to take into account"""
        X0=self.f
        Y0=self.FFT
        ind=X0>Fcut
        X=X0[ind]
        Y=Y0[ind]
        Npeak=Y.argmax()
        Fm=X[Npeak] #peak frequency
        self.Fpeak=Fm
        Width=np.zeros(len(Widthlevels))
        Fmean=np.zeros(len(Widthlevels))
        # StrLevels=''
        for i in range(len(Widthlevels)):
            if len(np.argwhere(Y[Npeak:]<Widthlevels[i]))==0:
                Nh=len(Y[Npeak:])-1
            else:
                Nh=np.argwhere(Y[Npeak:]<Widthlevels[i])[0]
            if Npeak == 0:
                Nl=0
            else:
                if len(np.argwhere(Y[:Npeak]<Widthlevels[i]))==0:
                    Nl=0
                else:
                    Nl=np.argwhere(Y[:Npeak]<Widthlevels[i])[-1]
            Width[i]=X[Npeak+Nh]-X[Nl]
            Fmean[i]=(X[Npeak+Nh]+X[Nl])/2
            # StrLevels+= ' '+str(Widthlevels[i])
            
        self.width=Width
        self.fmean=Fmean
        if self.usefiles:
            '/'.join(self.files[-1].split('/')[:-1])+'/AVfftparams.dat'
        else:
            file=self.folder+'/AVfftparams.dat'
        header='peak frequency Hz '+ '\t width at level' + str(Widthlevels[0])+ '\t width at level' + str(Widthlevels[1])+'\t fmean at level' + str(Widthlevels[0])+'\t fmean at level' + str(Widthlevels[1])
        np.savetxt(file, np.concatenate(([Fm],Width.reshape((1,-1))[0],Fmean.reshape((1,-1))[0])).reshape((1,-1)),
                        header=header,
                        delimiter='\t',comments='')
        self.average_output=pd.DataFrame([np.concatenate(([Fm],Width.reshape((1,-1))[0],Fmean.reshape((1,-1))[0]))],
                                         columns=header.split('\t'))
        
    def show_W(self,title=None,logscale=True):
        """plots the spectral signal"""
        if len(self.FFT) == 0:
            print('no spectral data run .FFT first')
        else:
            Y=self.FFT
            X=self.f
            ylable='amplitude (normalized)'
            xlable='frequency (Hz)'
            plt.plot(X,Y,linewidth=3)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel(xlable,fontsize=18)
            plt.ylabel(ylable,fontsize=18)
            if logscale:
                plt.yscale('log')
            if not title == None:
                plt.title(title,fontsize=18)
                
    def rescan_folder(self,outputPath = None):
        """go through the folder for th esecond time to save parameters for each """
        Params=[]
        for f in self.files:
            if self.usefiles:
                BS=Bird_sound_processor(f)
            else:
                BS=Bird_sound_processor(self.folder+'/'+f)
            BS.loadFFT()
            # print([self.fmean[0]-self.width[0]/2,self.fmean[0]+self.width[0]/2])
            Fband=[self.fmean[0]-self.width[0]/2,self.fmean[0]+self.width[0]/2]
            BS.find_main_Tslice(Fband)
            B=BS.export_params()
            Params.append(B[1])
            print('done with '+f)
        head=B[0]
        # print(Params)
        output=pd.DataFrame(Params,columns=head)
        if outputPath == None:
            if self.usefiles:
                Path=os.path.dirname((os.path.abspath(__file__)))
                outputPath=Path+'/output.csv'
            else:
                outputPath=self.folder+'/output.csv'
        output.to_csv(outputPath)
        self.output=output
            
    def run(self,output=None):
        """run functions to find parameters in a folder
        output is the output file path"""
        self.FFT_av()
        self.export_FFTparams()
        self.rescan_folder(outputPath=output)
        return [self.output,self.average_output]
                
    def show_phase(self,title):
        """show spectral phase"""
        pass

#test
# BS=Bird_sound_processor()

# # BS.convert_export()

# plt.figure(2)
# plt.clf()
# BS.FFT()
# BS.show_W()
# # Fband=[3737.4187444685203, 6357.615894039735]
# Fband=[2522.537919248024, 3897.5798821985536]
# BS.find_main_Tslice(Fband)


# plt.figure(1)
# plt.clf()
# BS.show_T()

# BS.export_params()




#folder

# BSF=sound_folder()
# BSF.FFT_av()
# plt.figure(3)
# plt.clf()
# BSF.show_W()
# BSF.export_FFTparams()
# BSF.rescan_folder()

# print([BSF.fmean[0]-BSF.width[0]/2,BSF.fmean[0]+BSF.width[0]/2])