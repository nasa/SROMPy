
import numpy as np
import matplotlib.pyplot as plt
import os


class Postprocessor:
    '''
    Class for comparing an SROM vs the target random vector it is modeling. 
    Capabilities for plotting CDFs/pdfs and tabulating errors in moments, 
    correlations, etc. 
    '''


    def __init__(self, srom, targetrv):
        '''
        Initialize class with previously initialized srom & targetrv objects.
        '''

        #TODO - check to make sure srom/targetrv are initialized & have the 
        #needed functions implemented (compute_moments/cdfs/ etc.)

        self._SROM = srom
        self._target = targetrv

    def compare_CDFs(self, variable="x", plotdir='.', plotsuffix="CDFcompare", 
                     showFig=True, saveFig=True):
        '''
        Generates plots comparing the srom & target cdfs for each dimension
        of the random vector.

        inputs:
            variable, str, name of variable being plotted
            plotsuffix, str, name for saving plot (will append dim & .pdf)
            plotdir, str, name of directory to store plots
            showFig, bool, show or not show generated plot
            saveFig, bool, save or not save generated plot

        '''

        xgrids = self.generate_cdf_grids()
        sromCDFs = self._SROM.compute_CDF(xgrids)
        targetCDFs = self._target.compute_CDF(xgrids)

        #Start plot name string if it's being stored
        if saveFig:
            plotname = os.path.join(plotdir, plotsuffix)
        else:
            plotname = None

        for i in range(self._SROM._dim):

            #Append dimension if this is random vector 
            if self._SROM._dim > 1:       
                variable += "_" + str(i+1)
                if saveFig:
                    plotname += "_d" + str(i+1) 
            if saveFig:
               plotname += ".pdf"

            ylabel = "F(" + variable + ")"

            self.plot_cdfs(xgrids[:,i], sromCDFs[:,i], targetCDFs[:,i], 
                           variable, ylabel, plotname, showFig)


    def plot_cdfs(self, xgrid, sromcdf, targetcdf, xlabel="x", ylabel="F(x)", 
                  plotname=None, showFig=True, xlimits=None):
        '''
        Plotting routine for comparing a single srom/target cdf
        '''
        
        #Text formatting for plot
        title_font = {'fontname':'Arial', 'size':22, 'weight':'bold',
                                    'verticalalignment':'bottom'}
        axis_font = {'fontname':'Arial', 'size':20, 'weight':'normal'}
        labelFont = 'Arial'
        labelSize = 18        

        #Plot CDFs
        fig,ax = plt.subplots(1)
        ax.plot(xgrid, sromcdf, 'r-', linewidth=2, label = 'SROM')
        ax.plot(xgrid, targetcdf, 'k--', linewidth=2, label = 'Target')
        ax.legend(loc='best', prop={'size':18})

        #Labels/limits    
        if(xlimits is not None):
            ax.axis([xlimits[0], xlimits[1], 0, 1])
        ax.set_xlabel(xlabel, **axis_font)
        ax.set_ylabel(ylabel, **axis_font)

        if plotname is not None:
            plt.savefig(plotname)
        if showFig:
            plt.show()    


    def generate_cdf_grids(self, cdf_grid_pts=500):
        '''
        Generate numerical grids for plotting CDFs based on the 
        range of the target random vector. Return  x_grid variable with
        cdf_grid_pts along each dimension of the random vector.
        '''

        x_grid = np.zeros((cdf_grid_pts, self._target._dim))

        for i in range(self._target._dim):
            grid = np.linspace(self._target._mins[i],
                               self._target._maxs[i],
                               cdf_grid_pts)
            x_grid[:,i] = grid

        return x_grid
