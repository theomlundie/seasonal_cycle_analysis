import numpy as np
import matplotlib.pylab as plt
import os
import datetime
import pandas as pd
import scipy.stats as sp
from scipy.optimize import curve_fit


# Choose usegas by unhashing:
usegas = 'ch4' ; gastext = 'CH$_4$'; unit = '(ppb)'; gasdir = 'ch4/'
#usegas = 'co2' ; gastext = 'CO$_2$'; unit = '(ppm)'; gasdir = 'co2/'
#usegas = 'co'  ; gastext = 'CO'; unit = '(ppb)'; gasdir = 'co/'
#usegas = 'c2h6'; gastext = 'C$_2$H$_6$'; unit = '(ppt)'; gasdir = 'c2h6/'


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def read_raw_daily_data(directory,filename):
    
    """
    This function reads discrete observational data from a directory+filename,
    applies relevant quality controls, and then returns arrays of
    concentration and time data.
    
    
    Inputs:
        
        directory - string corresponding to directory of file
        filename - string of file to be opened (including .txt suffix)
    
    Returns:
        Conc - numpy array of concentration values
        Time - numpy array of measurement times (float, years)
        Year - numpy array of measurement year (integer, years)
        Month - numpy array of measuremnet month (integer, month)

    """
    
    # return Conc, Time, Year, and Month lists converted to numpy arrays
    
    Time   = [] # create time list
    Conc    = [] # create gas conc list
    Year = [] # create year list
    Month = [] # create month list
    Day = [] # create day list



    f = open(directory+filename)

    
    tmp = f.readline()
    if tmp == '': tmp = f.readline()
    tmp = tmp.split(':') # Split first line
    
    nheader = int(tmp[1]) # Retrieve no. of header lines from first line
    for ii in np.arange(nheader-2): f.readline() # skipping header
    # Find variables from from final line of header:
    fvr = f.readline()
    fvr = fvr.split(' ')
    
    if usegas != 'c2h6' and usegas != 'c3h8':
        for jj in range(len(fvr)):
            
            if fvr[jj] == 'year' :
                ypos = jj
            elif fvr[jj] == 'month':
                mpos = jj
            elif fvr[jj] == 'day':
                dpos = jj
            elif fvr[jj] == 'hour':
                hpos = jj
            elif fvr[jj] == 'minute':
                minpos = jj
            elif fvr[jj] == 'value':
                vpos = jj
            elif fvr[jj] == 'qcflag\n':
                flagpos = jj
            elif fvr[jj] == 'latitude':
                latpos = jj
            elif fvr[jj] == 'longitude':
                longpos = jj
                
    elif usegas == 'c2h6' or usegas == 'c3h8': # if usegas is c2h6, use hard-coded values for variable positions
                ypos = 1
                mpos = 2
                dpos = 3
                hpos = 4
                minpos = 5
                vpos = 11
                flagpos = 13
                latpos = 21
                longpos = 22
                
        

    while 1: # infinite loop
        
        line = f.readline() # Read next line
        if not line: break # If can't read next line, break infinite loop
        
        cols = line.split() # split line into 'columns'
        
        # Assign positions of data columns for use
        flag = cols[flagpos]
        datestr = cols[ypos]+' '+cols[mpos]+' '+cols[dpos]+' '+cols[hpos]+' '+cols[minpos]
        d = datetime.datetime.strptime(datestr,'%Y %m %d %H %M').timetuple()
        TimeUse= d.tm_year + d.tm_yday/365.
        Lat = cols[latpos]
        Long = cols[longpos]

        #
        #   ... - No code applied. Data are considered 'background'
        #   *.. - Unable to compute a mole fraction or average
        #   I.. - No data available due to instrument calibration or malfunction.
        #   .V. - Large variability of CO2 mole fraction within one hour
        #   .D. - Hour-to-hour difference in msole fraction > 0.25 ppm
        #   .U. - Rejected, diurnal variation (upslope) in CO2 (Mauna Loa only)
        #   .S. - Single hour bracketed by flagged data.
        #   .N. - No unflagged data within +/- 8 hours. Assume non-background.
        #
        
        if flag == '...': # only write data to list if QC flag is good
            Time.append(TimeUse)
            Conc.append(float(cols[vpos]))
            Year.append(float(cols[ypos]))
            Month.append(float(cols[mpos]))
            Day.append(float(cols[dpos]))
            
            
        

    f.close() # close file
        
        
    # combine any data points on the same day and take their average
    
    data = pd.DataFrame({
    'year': Year,
    'month': Month,
    'day': Day,
    'time': Time,
    'concentration': Conc
    })
    
    # Group by year, month, and day and calculate the mean concentration for each group
    averaged_data = data.groupby(['year', 'month', 'day'], as_index=False).mean()
    
    # Convert the DataFrame columns back into separate lists
    avg_year = averaged_data['year'].tolist()
    avg_month = averaged_data['month'].tolist()
    avg_day = averaged_data['day'].tolist()
    avg_time = averaged_data['time'].to_list()
    avg_concentration = averaged_data['concentration'].tolist()
        
    # return as numpy arrays
    return np.array(avg_concentration), np.array(avg_time), np.array(avg_year), np.array(avg_month), Lat, Long


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def aggregated_percentile_method(nyears,minyear,maxyear,sst,time,months,timeint,stn):   
    """
    this function calculates the SCA across each calendar year by calculating
    the difference between the mean of the 0-10th percentile and 90-100th percentile
    of concentration in each calendar year that has sufficient data. 

    
    Inputs:
        nyears - float length of data record (years)
        minyear - float minimum dataset year
        maxyear - float maximum data year
        sst - numpy array of float species concentration
        time - numpy array of float time (years)
        months - numpy array of integer months (1-12)
        timeint - numpy array of integer years
        stn - string of station name
    
    Returns:
        minval - numpy array of float minimum seasonal cycle concentration values
        maxval - numpy array of float maximum seasonal cycle concentration values
        yearval - numpy array of float years where SCA determination is possible
        timeminval - numpy array of float minimum seasonal cycle value times
        timemaxval - numpy array of float maximum seasonal cycle value times
    """
    
    percentile = 10 # define percentile range from 0 and 100%
    
    
    # create lists to append values to
    
    minval     = [] # minimum value
    maxval     = [] # maximum value
    yearval    = [] # year
    timeminval = [] # time of minimum value
    timemaxval = [] # time of maximum value
    
    
    min_arr_years = []
    min_arr_inds = []
        

    max_arr_years = []
    max_arr_inds = []
    
    for ii in np.arange(nyears): # Loop through data years
           

        yearuse = int(minyear)+ii

        
        sel_ind = np.where(timeint == yearuse) # indices that fall within this year

        timeuse   = time[sel_ind] # times of data points within this year
        sstuse    = sst[sel_ind] # concentration values within this year
        
            
        
        # Apply checks on quantity and spread of available data within each year
        
        if not list(sstuse):
            print(yearuse, '*No data*')
        elif np.modf(np.min(timeuse))[0] > 0.1: # If data for this year doesn't start at the beginning
            print(yearuse, '*Not full data*')
        elif len(sstuse) < 12:
            print(yearuse, '*Not enough data*')
        else:
            
            
            
            
            minthresh = np.percentile(sstuse,percentile) # threshold for minima
            indmin = np.where(sstuse<=minthresh) # indices of concentrations below threshold
            
            
            min_arr_years.append(str(int(yearuse)))
            min_arr_inds.append(np.array(indmin))
            
            min_vals = sstuse[indmin] # all concs below threshold
            meanmin = np.mean(min_vals)
            minval.append(meanmin) # use bottom 10th percentile
            
            timeminval.append(np.mean(timeuse[indmin]))

            maxthresh = np.percentile(sstuse,(100-percentile)) # threshold for maxima
            indmax = np.where(sstuse>=maxthresh) # indices of concentrations above threshold
            
            max_arr_years.append(str(int(yearuse)))
            max_arr_inds.append(np.array(indmax))
            
            max_vals = sstuse[indmax] # all concs above threshold
            meanmax = np.mean(max_vals)
            maxval.append(meanmax) # use top 10th percentile
            
            timemaxval.append(np.mean(timeuse[indmax]))
            
            yearval.append(yearuse) # append year to list
      



    return minval, maxval, yearval, timeminval, timemaxval



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def remove_spikes_extra_functionality(conc, time, years, months, sitename, specstr):
    
    """
    This function reads takes c2h6 or c3h8 data, calculates a simple model of season cycle,
    filters data that aren't near this curve, and then returns the filtered data in same
    form as the input variables
    
    
    Inputs:
        conc - numpy array of species concentration
        time - numpy array of float times (years)
        years - numpy array of years (integers)
        months - numpy array of months (integers 1-12)
        sitename - string of sitename
        specstr - string of species type

    
    Returns:
        conc - numpy array of filtered concentration values
        time - numpy array of filtered  measurement times (float, years)
        years  - numpy array of measurement year (integer, years)
        months - numpy array of measuremnet month (integer, month)

    """

    # Define the model: sinusoidal oscillation + linear growth
    def model(time, A, f, phi, a, b):
        return A * np.sin(2 * np.pi * f * time + phi) + a + b * time

    # Initial parameter guesses
    
    amplitude = np.percentile(conc, 90) - np.percentile(conc, 10)
    frequency = 1 # 1 year period
    phase = -0.1 # peak should be jan/feb
    intercept = np.mean(conc) # mean conc for y axis intercept
    slope = 0 # no trend
    
    initial_guess = [amplitude, frequency, phase, intercept, slope]  

    # Fit the model to the data
    popt, pcov = curve_fit(model, time, conc, p0=initial_guess)

    # Extract fitted parameters
    A, f, phi, a, b = popt
    
    print()
    print(f"Fitted parameters:\nAmplitude (A): {A}\nFrequency (f): {f}\nPhase (phi): {phi}\nIntercept (a): {a}\nSlope (b): {b}")
    print()

    # Predict values using the fitted model
    y_fit = model(time, *popt)




    #~~~~~~~~ Plot the original data and the fit   
    
    plt.figure(figsize=(15, 9))
    
    plt.subplot(211)
    
    plt.plot(time, y_fit, label="Fitted Curve", color="darkgrey", linewidth=2)
    plt.scatter(time, conc, label="Data", color="blue", s=15)
    
    plt.xlabel("Time (years)")
    plt.ylabel("Molefraction "+unit)
    plt.title(sitename.upper()+" "+specstr+" Curve Fit: Annual Oscillations + Background Growth")
    plt.legend()
    plt.grid(True)
    
    
    plt.subplot(212)
    
    # assign the +/- tolerance margin as half the mean concentration across the data record
    margin = np.mean(conc) * 0.5

        
    
    difference = np.abs(conc-y_fit) # calculate difference between actual value and predicted values
    
    keep_ind = np.where(difference < margin)[0] # find indices of values to keep
    disc_ind = np.where(difference >= margin)[0] # find indices of values to discard
    
    dtime = time[disc_ind] # discarded times
    dconc = conc[disc_ind] # discarded concs

    
    ftime = np.arange(min(time), max(time), 0.002) # fill time for smooth fitting curve
    fconc = model(ftime, *popt) # fill concs for smooth fitting curve
    
    conc = conc[keep_ind] # kept concentrations
    time = time[keep_ind] # kept times
    

    # plot shaded area where data is kept
    plt.fill_between(ftime, fconc-margin, fconc+margin, color='gray', alpha = 0.3, label = 'bounds')
    
    # plot kept data, discarded data, and model fit
    plt.scatter(time, conc, label="Kept data", color="blue", s=15)
    plt.scatter(dtime, dconc, label="Discarded data", color="red", s=15)
    plt.plot(ftime, fconc, label="Fitted Curve", color="darkgrey", linewidth=2)
    
    
    plt.xlabel("Time (years)")
    plt.ylabel("Mole Fraction "+unit)
    
    plt.legend()
    plt.grid(True)
    
    plt.show()

    return conc, time, years, months


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def run_analysis(directory,filename): # data type, analysis technique, directory, filename
    
    """
    this function coordinates the SCA analysis and plotting for the file
    specified in directory and filename input variables.
    
    Inputs:
        directory - string corresponding to directory of file
        filename - string of file to be opened (including .txt suffix)
    
    Returns:
    """

    if filename.endswith('txt'): station = filename.split('_')[1]
    else: print('file must be .txt format'); return

    
    
    # read in concnetration, time, year and month data from file    
    conc, all_time, all_years, all_months, latitude, longitude = read_raw_daily_data(directory, filename)
            

    if usegas == 'c2h6' or usegas == 'c3h8':
        # if usegas is c2h6 or c3h8 then filter data that are anomalously erratic
        conc, all_time, all_years, all_months = remove_spikes_extra_functionality(conc, all_time, all_years, all_months, station, usegas)




    #~~~~~~~~ Do SCA analysis & plot       
        
    PlotTitle = station.upper()+' '+str(latitude)+'°N , '+str(longitude)+'°E'
    
    
    sst      = conc
    time     = np.array(all_time) # Convert time list to NumPy array
    months   = np.array(all_months) # Convert months list to NumPy array
    years    = np.array(all_years) # Convert years list to NumPy array
    timeint  = time.astype(int) # Convert time list to integer years


    # Find min and max time of data points
    minyear = np.min(time)
    maxyear = np.max(time)
    
    # Fit x-axes to correct time interval
    plotyearmin = 5*np.floor(minyear/5) 
    plotyearmax = 5*np.ceil(maxyear/5)
    
    # Alternatively, fit x-axes to set time interval (1980-2025)
    #plotyearmin = 1980
    #plotyearmax = 2025

        
    nyears = maxyear - minyear # Calculate total time period
    
    if nyears < 10: return # exit if less than 10 years of data available


    # calculate SCA maxima and minima with associated times
    minval, maxval, yearval, timeminval, timemaxval = aggregated_percentile_method(nyears,minyear,maxyear,sst,time,months,timeint,station)
        

    plt.figure(1,figsize=(8,8))
    
    plt.subplot(211) # first plot panel
    
    plt.plot(time,sst,'o',color='darkgray',alpha=0.75,markersize=3,label='CH$_4$ measurement')
    plt.plot(timemaxval,maxval,'go',label='Maxima')
    plt.plot(timeminval,minval,'bo',label='Minima')
    plt.title(PlotTitle)
    plt.xlim([plotyearmin,plotyearmax])
    
    plt.ylabel(gastext+' Mole Fraction '+unit)
    
    plt.xlabel('Year')
    

    plt.legend() # display legend

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    plt.subplot(212) # second plot panel
    
    yearvalshift = np.add(yearval,0.5) # shift times by half a year for plotting
    
    amplitude = np.subtract(maxval,minval) # calcualte amplitudes as maxima minus minima
    


    plt.plot(yearval,amplitude,'ro')#,label='Amplitude') 
    plt.xlim([plotyearmin,plotyearmax])
    plt.ylabel('Seasonal Cycle Amplitude '+unit)
    plt.xlabel('Year')       

    # Theil slopes analysis for SCA trend
    theil_out = sp.mstats.theilslopes(amplitude,yearvalshift,alpha=0.95)    
    print('************', theil_out[0],theil_out[1],theil_out[2]-theil_out[0])
    tgrad = theil_out[0] # theil gradient
    intcpt = theil_out[1] # y intercept
    lowslope = theil_out[2] # lower bound of slope confidence interval
    upslope = theil_out[3] # upper bound of slope confidence interval
    tx = yearvalshift
    ty = np.multiply(tx,tgrad) +intcpt
    bounds_str = ' ['+str(np.around(lowslope,3))+', '+str(np.around(upslope,3))+']'
    theil_label = 'Theil-Sen gradient = '+ str(np.around(tgrad,3))+bounds_str+' '+unit[1:-1]+' yr$^-$$^1$'
    plt.plot(tx,ty,'g',ls='--',label = theil_label) # plot Theil-Sen trend


    # Linear polynomial analysis for SCA trend
    output, V = np.polyfit(yearvalshift,amplitude,deg=1,cov=True)
    x = yearvalshift # assign x values
    y = np.multiply(x,output[0]) + output[1] # calculate y = mx + c
    lgrad = "{:.3f}".format(float(output[0])) # convert linear fit gradient to 3dp str
    lstd = "{:.3f}".format(float(np.sqrt(V[0][0]))) # convert linear fit standard deviation to 3dp str
    linear_label = 'Linear polynomial gradient = '+lgrad+'$\pm$'+lstd+' '+unit[1:-1]+' yr$^-$$^1$'
    plt.plot(x,y,'r',label=linear_label) # plot linear trend
        

    plt.legend() # display legend
    
    
    plt.tight_layout()

    
    
    plt.show() # show plot




# MAIN BODY

if __name__=='__main__':
    
    
    use_files = [] # create list to append files to
    
    for file in os.listdir(gasdir):
        
        if file.endswith('event.txt'):
            
            use_files.append(file)
            
    for file in use_files:
        
        run_analysis(gasdir, file)
   
    print('Program Finished')
    
    
    
    
    