
from collections import OrderedDict
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
                     showFig=True, saveFig=True, variablenames=None, 
                     xlimits=None, ylimits=None, xticks=None,
                     cdfylabel=False, xaxispadding=None,
                     axisfontsize=30, labelfontsize=24,
                     legendfontsize=25):
        '''
        Generates plots comparing the srom & target cdfs for each dimension
        of the random vector.

        inputs:
            variable, str, name of variable being plotted
            plotsuffix, str, name for saving plot (will append dim & .pdf)
            plotdir, str, name of directory to store plots
            showFig, bool, show or not show generated plot
            saveFig, bool, save or not save generated plot
            variablenames, list of strings, names of variable in each dimension
                optional. Used for x axes labels if provided. 
        '''

        xgrids = self.generate_cdf_grids()
        sromCDFs = self._SROM.compute_CDF(xgrids)
        targetCDFs = self._target.compute_CDF(xgrids)

        #Start plot name string if it's being stored
        if saveFig:
            plotname = os.path.join(plotdir, plotsuffix)
        else:
            plotname = None

        #Get variable names:
        if variablenames is not None:
            if len(variablenames) != self._SROM._dim:
                raise ValueError("Wrong number of variable names provided")
        else:
            variablenames = []
            for i in range(self._SROM._dim):
                if self._SROM._dim==1:
                    variablenames.append(variable)
                else:
                    variablenames.append(variable + "_" + str(i+1))

        for i in range(self._SROM._dim):

            variable = variablenames[i]
            ylabel = "F(" + variable + ")"
            #Remove latex math symbol from plot name
            if plotname is not None:
                plotname_ = plotname + "_" + variable + ".pdf"
                plotname_ = plotname_.translate(None, "$")
            else:
                plotname_ = None
            if xlimits is not None:
                xlims = xlimits[i]
            else:
                xlims = None
            self.plot_cdfs(xgrids[:,i], sromCDFs[:,i], xgrids[:,i], 
                targetCDFs[:,i],  variable, ylabel, plotname_, showFig, xlims)


    def compare_pdfs(self, variable="x", plotdir='.', plotsuffix="pdf_compare", 
                     showFig=True, saveFig=True, variablenames=None, 
                     xlimits=None, ylimits=None, xticks=None,
                     cdfylabel=False, xaxispadding=None,
                     axisfontsize=30, labelfontsize=24,
                     legendfontsize=25):
        '''
        Generates plots comparing the srom & target pdfs for each dimension
        of the random vector.

        inputs:
            variable, str, name of variable being plotted
            plotsuffix, str, name for saving plot (will append dim & .pdf)
            plotdir, str, name of directory to store plots
            showFig, bool, show or not show generated plot
            saveFig, bool, save or not save generated plot
            variablenames, list of strings, names of variable in each dimension
                optional. Used for x axes labels if provided. 
        '''

        xgrids = self.generate_cdf_grids()
        targetCDFs = self._target.compute_pdf(xgrids)

        (samples, probs) = self._SROM.get_params()

        #Start plot name string if it's being stored
        if saveFig:
            plotname = os.path.join(plotdir, plotsuffix)
        else:
            plotname = None

        #Get variable names:
        if variablenames is not None:
            if len(variablenames) != self._SROM._dim:
                raise ValueError("Wrong number of variable names provided")
        else:
            variablenames = []
            for i in range(self._SROM._dim):
                if self._SROM._dim==1:
                    variablenames.append(variable)
                else:
                    variablenames.append(variable + "_" + str(i+1))

        if len(samples.shape)==1:
            samples = samples.reshape((1, len(samples)))

        for i in range(self._SROM._dim):

            variable = variablenames[i]
            ylabel = "f(" + variable + ")"
            #Remove latex math symbol from plot name
            if plotname is not None:
                plotname_ = plotname + "_" + variable + ".pdf"
                plotname_ = plotname_.translate(None, "$")
            else:
                plotname_ = None
            if xlimits is not None:
                xlims = xlimits[i]
            else:
                xlims = None
            print "samples = ", samples[:,i]
            self.plot_pdfs(samples[:, i], probs.flatten(),  xgrids[:,i], 
                targetCDFs[:,i],  variable, ylabel, plotname_, showFig, xlims)


    def compute_moment_error(self, max_moment=4):
        '''
        Performs a comparison of the moments between the SROM and target, 
        calculates the percent errors up to moments of order 'max_moment'.
        Optionally generates text file with the latex source to generate
        a table.
        '''

        #Get moment arrays (max_order x dim)
        srom_moments = self._SROM.compute_moments(max_moment)   
        target_moments = self._target.compute_moments(max_moment)   
        
        percent_errors = np.abs(srom_moments-target_moments)
        percent_errors = percent_errors/np.abs(target_moments)

        return percent_errors*100


    def plot_cdfs(self, xgrid, sromcdf, xtarget, targetcdf, xlabel="x", 
                 ylabel="F(x)",  plotname=None, showFig=True, xlimits=None):
        '''
        Plotting routine for comparing a single srom/target cdf
        '''
        
        #Text formatting for plot
        title_font = {'fontname':'Arial', 'size':22, 'weight':'bold',
                                    'verticalalignment':'bottom'}
        axis_font = {'fontname':'Arial', 'size':26, 'weight':'normal'}
        labelFont = 'Arial'
        labelSize =  20      
        legendFont = 22
    
        #Plot CDFs
        fig, ax = plt.subplots(1)
        ax.plot(xgrid, sromcdf, 'r--', linewidth=4.5, label = 'SROM')
        ax.plot(xtarget, targetcdf, 'k-', linewidth=2.5, label = 'Target')
        ax.legend(loc='best', prop={'size': legendFont})
        fig.canvas.set_window_title("CDF Comparison")

        #Labels/limits    
        y_limz = ax.get_ylim()
        x_limz = ax.get_xlim()
        ax.axis([min(xgrid), max(xgrid), 0, 1.1])
        if(xlimits is not None):
            ax.axis([xlimits[0], xlimits[1], 0, 1.1])
        ax.set_xlabel(xlabel, **axis_font)
        ax.set_ylabel(ylabel, **axis_font)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname(labelFont)
            label.set_fontsize(labelSize)

        #NOTE TMP HACK FOR PHM18
        #xticks = ['', r'$1.2 \times 10^6$','',r'$1.6 \times 10^6$','',
        #   r'$2.0 \times 10^6$','']
        #ax.set_xticklabels(xticks)       

        plt.tight_layout()

        if plotname is not None:
            plt.savefig(plotname)
        if showFig:
            plt.show()    


    def plot_pdfs(self, samples, probs, xtarget, targetpdf, xlabel="x", 
                 ylabel="f(x)",  plotname=None, showFig=True, xlimits=None):
        '''
        Plotting routine for comparing a single srom/target pdf
        '''
        
        #Text formatting for plot
        title_font = {'fontname':'Arial', 'size':22, 'weight':'bold',
                                    'verticalalignment':'bottom'}
        axis_font = {'fontname':'Arial', 'size':26, 'weight':'normal'}
        labelFont = 'Arial'
        labelSize =  20      
        legendFont = 22
    
        #Scale SROM probs such that max(SROM_prob) = max(target_prob)
        scale = max(targetpdf) / max(probs)
        probs *= scale

        #Get width of bars in some intelligent way:
        xlen = max(xtarget) - min(xtarget)
        width = 0.1*xlen/len(samples)  #bars take up 10% of x axis? 

        #Plot PDFs
        fig, ax = plt.subplots(1)
        ax.plot(xtarget, targetpdf, 'k-', linewidth=2.5, label = 'Target')
        ax.bar(samples, probs, width, color='red', label='SROM')

        ax.legend(loc='best', prop={'size': legendFont})
        fig.canvas.set_window_title("PDF Comparison")

        #Labels/limits    
        y_limz = ax.get_ylim()
        x_limz = ax.get_xlim()
#        ax.axis([min(xtarget), max(xtarget), 0, 1.1])
        if(xlimits is not None):
            ax.axis([xlimits[0], xlimits[1], 0, 1.1])
        ax.set_xlabel(xlabel, **axis_font)
        ax.set_ylabel(ylabel, **axis_font)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname(labelFont)
            label.set_fontsize(labelSize)

        plt.tight_layout()

        if plotname is not None:
            plt.savefig(plotname)
        if showFig:
            plt.show()    


    def generate_cdf_grids(self, cdf_grid_pts=1000):
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



#Need to make this whole class static
#----------------Gross down here, don't look -----------------------
    @staticmethod
    def compare_srom_CDFs(size2srom, target, variable="x", plotdir=".",
                            plotsuffix="CDFscompare", showFig=True, 
                            saveFig = True, variablenames=None,
                            xlimits=None, ylimits=None, xticks=None,
                            cdfylabel=False, xaxispadding=None, 
                            axisfontsize=30, labelfontsize=24,
                            legendfontsize=25):
        '''
        Generates plots comparing CDFs from sroms of different sizes versus 
        the target variable for each dimension of the vector.

        inputs:
            size2srom, dict, key=size of SROM (int), value = srom object
            target, TargetRV, target random variable object
            variable, str, name of variable being plotted
            plotsuffix, str, name for saving plot (will append dim & .pdf)
            plotdir, str, name of directory to store plots
            showFig, bool, show or not show generated plot
            saveFig, bool, save or not save generated plot
            variablenames, list of strings, names of variable in each dimension
                optional. Used for x axes labels if provided. 
            cdfylabel, bool, use "CDF" as y-axis label? If False, uses 
                            F(<variable_name>)
            xaxispadding, int, spacing between xtick labels and x-axis
        '''

        #Make x grids for plotting
        cdf_grid_pts=1000
        xgrids = np.zeros((cdf_grid_pts, target._dim))

        for i in range(target._dim):
            grid = np.linspace(target._mins[i],
                               target._maxs[i],
                               cdf_grid_pts)
            xgrids[:,i] = grid

        #If xticks is None (default), cast to list of Nones for plotting:
        if xticks is None:
            xticks = target._dim * [None]

        #Get CDFs for each size SROM
        sromCDFs = OrderedDict()
        for m,srom in size2srom.iteritems():
            sromCDFs[m] = srom.compute_CDF(xgrids)

        #targetCDFs = target.compute_CDF(xgrids)
        (target_grids, targetCDFs) = target.get_plot_CDFs()


        #Start plot name string if it's being stored
        if saveFig:
            plotname = os.path.join(plotdir, plotsuffix)
        else:
            plotname = None

        #Get variable names:
        if variablenames is not None:
            if len(variablenames) != target._dim:
                raise ValueError("Wrong number of variable names provided")
        else:
            variablenames = []
            for i in range(target._dim):
                if target._dim==1:
                    variablenames.append(variable)
                else:
                    variablenames.append(variable + "_" + str(i+1))

        for i in range(target._dim):

            variable = variablenames[i]
            plotname_ = plotname + "_" + variable + ".pdf"
            if not cdfylabel:
                ylabel = r'$F($' + variable + r'$)$'
            else:
                ylabel = "CDF"             
   
            #Remove latex math symbol from plot name
            plotname_ = plotname_.translate(None, "$")

            #---------------------------------------------
            #PLOT THIS DIMENSION:
            #Text formatting for plot
            title_font = {'fontname':'Arial', 'size':22, 'weight':'bold',
                                    'verticalalignment':'bottom'}
            axis_font = {'fontname':'Arial', 'size':axisfontsize, 
                         'weight':'normal'}
            labelFont = 'Arial'
            labelSize =  labelfontsize
            legendFont = legendfontsize
       
            xgrid=xgrids[:,i]
            xtarget = target_grids[:,i]
            targetcdf = targetCDFs[:,i]

            linez = ['g-','r:','b--']
            widthz = [2.5, 4, 3.5]
            #Plot CDFs
            fig,ax = plt.subplots(1)
            ax.plot(xtarget, targetcdf, 'k-', linewidth=2.5, label = 'Target')
            for j,m in enumerate(sromCDFs.keys()):
                #label = "SROM (m=" + str(m) + ")"
                label = "m = " + str(m) 
                sromcdf = sromCDFs[m][:,i]
                ax.plot(xgrid, sromcdf, linez[j], linewidth=widthz[j], label=label)
            ax.legend(loc='best', prop={'size': legendFont})

            #Labels/limits    
            y_limz = ax.get_ylim()
            x_limz = ax.get_xlim()
            ax.axis([min(xgrid), max(xgrid), 0, 1.1])
            if (xlimits is not None and ylimits is None):
                ax.axis([xlimits[i][0], xlimits[i][1], 0, 1.1])
            elif (xlimits is None and ylimits is not None): 
                ax.axis([x_limz[0], x_limz[1], ylimits[i][0], ylimits[i][1]])
            elif (xlimits is not None and ylimits is not None): 
                ax.axis([xlimits[i][0], xlimits[i][1], 
                         ylimits[i][0], ylimits[i][1]])
            else:
                ax.axis([min(xgrid), max(xgrid), 0, 1.1])

            ax.set_xlabel(variable, **axis_font)
            ax.set_ylabel(ylabel, **axis_font)

            if xaxispadding:
                ax.tick_params(axis='x', which='major', pad=xaxispadding)

            #Adjust tick labels:    
            if xticks[i] is not None:
                ax.set_xticklabels(xticks[i])

            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontname(labelFont)
                label.set_fontsize(labelSize)

            plt.tight_layout()

            if plotname_ is not None:
                plt.savefig(plotname_)
            if showFig:
                plt.show()

    @staticmethod
    def compare_RV_CDFs(RV1, RV2, variable="x", plotdir=".",
                            plotsuffix="CDFscompare", showFig=True, 
                            saveFig = False, variablenames=None,
                            xlimits=None, labels=None):
        '''
        Generates plots comparing CDFs from sroms of different sizes versus 
        the target variable for each dimension of the vector.

        inputs:
            RV1, SampleRandomVector, target random variable object
            RV2, SampleRandomVector, target random variable object
            variable, str, name of variable being plotted
            plotsuffix, str, name for saving plot (will append dim & .pdf)
            plotdir, str, name of directory to store plots
            showFig, bool, show or not show generated plot
            saveFig, bool, save or not save generated plot
            variablenames, list of strings, names of variable in each dimension
                optional. Used for x axes labels if provided. 
            labels, list of str: names of RV1 & RV2
        '''

        (rv1_grids, rv1CDFs) = RV1.get_plot_CDFs()
        (rv2_grids, rv2CDFs) = RV2.get_plot_CDFs()

        #Start plot name string if it's being stored
        if saveFig:
            plotname = os.path.join(plotdir, plotsuffix)
        else:
            plotname = None

        #Get variable names:
        if variablenames is not None:
            if len(variablenames) != target._dim:
                raise ValueError("Wrong number of variable names provided")
        else:
            variablenames = []
            for i in range(RV1._dim):
                if RV1._dim==1:
                    variablenames.append(variable)
                else:
                    variablenames.append(variable + "_" + str(i+1))

        print "variable names = ", variablenames

        for i in range(RV1._dim):

            variable = variablenames[i]
            if plotname is not None:
                plotname_ = plotname + "_" + variable + ".pdf"
                #Remove latex math symbol from plot name
                plotname_ = plotname_.translate(None, "$")
            else:
                plotname_ = None

            ylabel = r'$F($' + variable + r'$)$'

            #---------------------------------------------
            #PLOT THIS DIMENSION:
            #Text formatting for plot
            title_font = {'fontname':'Arial', 'size':22, 'weight':'bold',
                                    'verticalalignment':'bottom'}
            axis_font = {'fontname':'Arial', 'size':26, 'weight':'normal'}
            labelFont = 'Arial'
            labelSize =  20
            legendFont = 22
       
            x1 = rv1_grids[:,i]
            cdf1 = rv1CDFs[:,i]
            x2 = rv2_grids[:,i]
            cdf2 = rv2CDFs[:,i]

            linez = ['g-','r:','b--']
            widthz = [2.5, 4, 3.5]
            #Plot CDFs
            fig,ax = plt.subplots(1)
            if labels is None:
                labels = ["RV1", "RV2"]

            ax.plot(x1, cdf1, 'r--', linewidth=4.5, label=labels[0])
            ax.plot(x2, cdf2, 'b-', linewidth=2.5, label=labels[1])

            ax.legend(loc='best', prop={'size': legendFont})

            #Labels/limits    
            y_limz = ax.get_ylim()
            x_limz = ax.get_xlim()
            ax.axis([min(x1), max(x1), 0, 1.1])
            if(xlimits is not None):
                ax.axis([xlimits[i][0], xlimits[i][1], 0, 1.1])
            ax.set_xlabel(variable, **axis_font)
            ax.set_ylabel(ylabel, **axis_font)

            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontname(labelFont)
                label.set_fontsize(labelSize)

            plt.tight_layout()

            if plotname_ is not None:
                plt.savefig(plotname_)
            if showFig:
                plt.show()

