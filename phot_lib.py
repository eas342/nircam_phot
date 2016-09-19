from astropy.io import fits
import numpy as np
import glob
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
import os
import matplotlib.pyplot as plt
from astropy.table import hstack, vstack, Table
from astropy.time import Time
import pdb
from astropy.modeling import models, fitting
import re
import es_gen

def get_slope_img(file,slopeType='Last - First'):
    """
    Gets the slope image from a file.
    slopeType can be 'Last-First' for last frame minus first 
                                   divided by the integration time - group time
                     'Slope File' for just the first plane
                     'Last Only' for just the last frame divided by the int time
    """
    HDU = fits.open(file)
    data = HDU[0].data
    header = HDU[0].header
    if slopeType == 'Last - First':
        diff = data[-1,:,:] - data[0,:,:]
        time = header['INTTIME'] - header['TGROUP']
        slp = diff / time
    elif slopeType == 'Slope File':
        slp = data[0,:,:]
    elif slopeType == 'Last Only':
        slp = data[-1,:,:]
        time = header['INTTIME']
        slp = diff / time
    else:
        print('Invalid Slope Type')
        slp = np.array(0)
    HDU.close()
    return slp, header

def find_gausspeak(img,posGuess,widthGuess=2,windowSize=10,showPlot=False):
    """ Finds the Peak with a Gaussian 2D fit"""
    
    y, x = np.mgrid[0:img.shape[0],0:img.shape[1]]
	
    ## Get the sub-image around the point
    xst, xend = int(posGuess[0] - windowSize/2.), int(posGuess[0] + windowSize/2.)
    yst, yend = int(posGuess[1] - windowSize/2.), int(posGuess[1] + windowSize/2.)
    xsub = x[yst:yend,xst:xend]
    ysub = y[yst:yend,xst:xend]
    subimg = img[yst:yend,xst:xend]

    ## Fit with a Gaussian
    fit_g = fitting.LevMarLSQFitter()
    ampguess = np.nanpercentile(img,99.99)
    g_init = models.Gaussian2D(amplitude=ampguess, x_mean=posGuess[0], y_mean=posGuess[1],
                               x_stddev=widthGuess,y_stddev=widthGuess)
    result = fit_g(g_init,xsub,ysub,subimg)

    if showPlot:
        ## Show plots
        modelimg = result(xsub,ysub)
        residimg = subimg - modelimg
        fig, ax = plt.subplots(1,3)
        ax[0].imshow(subimg)
        ax[0].set_title('Data')
        ax[1].imshow(modelimg)
        ax[1].set_title('Model')
        ax[2].imshow(residimg)
        ax[2].set_title('Residual')
    
    return [result.x_mean.value, result.y_mean.value], result.amplitude.value

def do_phot(fileName,apPos=[1406,1039],r_src=70,r_in=72,r_out=80,showPlot=False,
            slopeType='Last - First',fixedPosition=False):
    img, header = get_slope_img(fileName,slopeType=slopeType)
    apGuess = apPos - np.array([header['COLCORNR'],header['ROWCORNR']])
    if fixedPosition:
        apUse = apGuess
        ampPeak = np.percentile(img,99.999)
    else:
        apUse, ampPeak = find_gausspeak(img,apGuess)
        
    apSource = CircularAperture(apUse,r_src)
    apBack = CircularAnnulus(apUse,r_in=r_in,r_out=r_out)
    
    ## Do the photometry
    mask = np.isnan(img)
    photTable = aperture_photometry(img, apSource,mask=mask)
    backTable = aperture_photometry(img, apBack,mask=mask)
    photTable = hstack([photTable, backTable], table_names=['raw', 'bkg'])
    photTable['resid_sum'] = (photTable['aperture_sum_raw'] - 
                              photTable['aperture_sum_bkg'] * apSource.area() / apBack.area())
    photTable['file name'] = os.path.basename(fileName)

    ## Find the time
    timeRel = (header['ON_NINT'] * (header['INTTIME']+header['TGROUP']))/(24. * 3600.)
    timeRef = Time(header['DATE-OBS']+'T'+header['TIME-OBS']).jd
    tstart = timeRel + timeRef
    photTable['time-start'] = tstart
    
    ## Plot
    if showPlot:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(img,vmin=0,vmax=1.0 * ampPeak)
        apSource.plot(ax=ax,color='white',linewidth=3)
        apBack.plot(ax=ax,color='yellow',linewidth=3)
        ax.set_xlim(apUse[0] - r_out - 30, apUse[0] + r_out + 30)
        ax.set_ylim(apUse[1] - r_out - 30, apUse[1] + r_out + 30)
        fig.savefig('phot_aps.pdf')

    photTable['x_used'] = photTable['xcenter_raw'].data[0]
    photTable['y_used'] = photTable['ycenter_raw'].data[0]
    photTable['x_absolute'] = photTable['x_used'] + header['COLCORNR']
    photTable['y_absolute'] = photTable['y_used'] + header['ROWCORNR']
    keepParams = ['aperture_sum_raw','resid_sum','aperture_sum_bkg','file name',
                  'time-start','x_used','y_used','x_absolute','y_absolute']

    return photTable[keepParams]

def get_file_table(testDirectories,fileType='.red',allFiles=False):
    fileList, testList, slopeTypeList = [], [], []
    ## Ignore this test since these files were corrupted
    badDirectories = ['_2016-01-12T17h09m18','_2016-01-12T17h13m56']
    for testDir in testDirectories:
        isGoodDirectory = True
        for badDirectory in badDirectories:
            if badDirectory in testDir:
                isGoodDirectory = False
        if isGoodDirectory:
            testName = os.path.basename(testDir).split('-')[0].split('NRCN821')[-1]
            if fileType == 'slp/red':
                if 'SUB' in testName:
                    useFileType = '.slp'
                    SlopeType = 'Slope File'
                else:
                    useFileType = '.red'
                    SlopeType = 'Last - First'
            else:
                useFileType = fileType
                SlopeType = 'Last - First'

            if allFiles == True:
                fileSearchShort = testDir+'/NRCN*'+useFileType+'.fits'
                fileSearchLong = ''
            else:
                fileSearchShort = testDir+'/*I51'+useFileType+'.fits'
                fileSearchLong = testDir+'/*I051'+useFileType+'.fits'
                # These two searches are necessary to find a specific file
                # since the integrations are split into I051 and I51 depending
                # on the number of integrations

            for search in [fileSearchLong,fileSearchShort]:
                newFiles = glob.glob(search)

                fileList = fileList + newFiles
                testList = testList + [testName] * len(newFiles)
                slopeTypeList = slopeTypeList + [SlopeType] * len(newFiles)
    fileTable = Table()
    fileTable['Full Path'] = fileList
    fileTable['Test Name'] = testList
    fileTable['Slope Type'] = slopeTypeList
    return fileTable

def get_phot_table(fileTable,name='phot',**kwargs):
                   
    t = Table()
    for file in fileTable:
        phot_table = do_phot(file['Full Path'],slopeType=file['Slope Type'],
                             **kwargs)
        phot_table['Test Name'] = file['Test Name']
        t = vstack([t,phot_table])


    fluxRef = np.nanmedian(t['resid_sum'])
    t['norm_flux'] = t['resid_sum'] / fluxRef
    t['Full Path'] = fileTable['Full Path']

    # Save to file
    t.write('output_data/longWLP8_series/tser_'+name+'.csv')
    return t

def plot_long_series(t,fileType,ax=None,fig=None,combo=False,
                     returnFigs=False):
    if fig == None:
        fig, ax = plt.subplots(figsize=(15,5))

    if combo==True:
        testNameUse = 'Test Name_1'
        timeUse = 'time-start_1'
        fluxUse = 'avg_flux'
        fluxLabel = 'Flux (DN/s)'
    else:
        testNameUse = 'Test Name'
        timeUse = 'time-start'
        fluxUse = 'norm_flux'

    uniqTest = np.unique(t[testNameUse])
    for test in uniqTest:
        testp = t[testNameUse] == test
        x = t[timeUse][testp]
        y = t[fluxUse][testp]
        ax.plot(x,y,'o')
        ax.text(np.nanmean(x),np.nanmin(y) - 1e-3,test,horizontalalignment='center',
                verticalalignment='top')
    ax.set_title('Using '+fileType)
    ax.set_xlabel('Time (JD)')
    ax.set_ylabel('Normalized Flux')
    if returnFigs == True:
        return fig, ax

def several_test_series(testDirectories,testName='Group',multiType='All',
                        guessCoord = [1406,1039],**kwargs):
    """ Collects the file table, photometry and makes a plot for a long time series"""
    if multiType == 'All':
        fileTypes = ['.red','slp/red','']
        fileTypeNames = ['.red','slp/red','raw']
    else:
        fileTypes = ['slp/red']
        fileTypeNames = ['slp/red']

    fig, axesAll = plt.subplots(len(fileTypes),1,figsize=(15,5 * len(fileTypes)))
    axes = fig.axes
    for ind, fileType in enumerate(fileTypes):
        fileTable = get_file_table(testDirectories,fileType=fileType)
        ## Make the file name cleaner
        cleanName = '_'.join(re.split(r'\.|/',fileTypeNames[ind]))
        t = get_phot_table(fileTable,name=testName+'_'+cleanName,apPos=guessCoord,
                           **kwargs)
        plot_long_series(t,fileTypeNames[ind],ax=axes[ind],fig=fig)

    fig.savefig('plots/tser_'+testName+'.pdf')
    return t

def comb_phot(photTab,table_names=None,avgFlux=True):
    """ Combines photometry Tables - usually the A and B sides """

    ## Find the lengths of the tables and get the minimum
    lenArr = []
    for oneTab in photTab:
        lenArr.append(len(oneTab))
    useLen = np.min(lenArr)

    ## Truncate all tables to the common length and find average flux
    useArr = []
    totFlux = np.zeros(useLen)
    for oneTab in photTab:
        shrinkTable = oneTab[0:useLen]
        useArr.append(shrinkTable)
        if avgFlux == True:
            totFlux = totFlux + shrinkTable['norm_flux']

    comb = hstack(useArr,table_names=table_names)
    if avgFlux == True:
        avgFlux = totFlux / float(len(photTab))
        comb['avg_norm_fl'] = avgFlux

        x = useArr[0]['time-start']
        polyCoeff = es_gen.robust_poly(x,comb['avg_norm_fl'],1)
        comb['avg_detrend_fl'] = avgFlux / np.polyval(polyCoeff,x)

    return comb


