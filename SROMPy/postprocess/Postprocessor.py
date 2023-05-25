# Copyright 2018 United States Government as represented by the Administrator of
# the National Aeronautics and Space Administration. No copyright is claimed in
# the United States under Title 17, U.S. Code. All Other Rights Reserved.

# The Stochastic Reduced Order Models with Python (SROMPy) platform is licensed
# under the Apache License, Version 2.0 (the "License"); you may not use this
# file except in compliance with the License. You may obtain a copy of the
# License at http://www.apache.org/licenses/LICENSE-2.0.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import os

from SROMPy.srom.SROM import SROM
from SROMPy.target.RandomEntity import RandomEntity


class Postprocessor:
    """
    Class for comparing an SROM vs the target random vector it is modeling. 
    Capabilities for plotting CDFs/pdfs and tabulating errors in moments, 
    correlations, etc. 
    """

    def __init__(self, srom, target_random_vector):
        """
        Initialize class with previously initialized srom & targetrv objects.
        """
        self.__check_init_params(srom, target_random_vector)

        self._SROM = srom
        self._target = target_random_vector

    def compare_cdfs(self, variable="x", plot_dir='.', plot_suffix="CDFcompare",
                     show_figure=True, save_figure=True, variable_names=None,
                     x_limits=None):
        """
        Generates plots comparing the srom & target cdfs for each dimension
        of the random vector.

        inputs:
            variable, str, name of variable being plotted
            plot_suffix, str, name for saving plot (will append dim & .pdf)
            plot_dir, str, name of directory to store plots
            show_figure, bool, show or not show generated plot
            save_figure, bool, save or not save generated plot
            variable_names, list of strings, names of variable in each dimension
                optional. Used for x axes labels if provided. 
        """

        x_grids = self.generate_cdf_grids()
        srom_cdfs = self._SROM.compute_cdf(x_grids)
        target_cdfs = self._target.compute_cdf(x_grids)

        # Start plot name string if it's being stored.
        if save_figure:
            plot_name = os.path.join(plot_dir, plot_suffix)
        else:
            plot_name = None

        # Get variable names:
        if variable_names is not None:
            if len(variable_names) != self._SROM._dim:
                raise ValueError("Wrong number of variable names provided")
        else:
            variable_names = []
            for i in range(self._SROM._dim):
                if self._SROM._dim == 1:
                    variable_names.append(variable)
                else:
                    variable_names.append(variable + "_" + str(i + 1))

        for i in range(self._SROM._dim):

            variable = variable_names[i]
            y_label = "F(" + variable + ")"

            # Remove latex math symbol from plot name
            if plot_name is not None:
                plot_name_ = plot_name + "_" + variable + ".pdf"
                # plot_name_ = plot_name_.translate(None, "$")
            else:
                plot_name_ = None
            if x_limits is not None:
                x_limit = x_limits[i]
            else:
                x_limit = None
            self.plot_cdfs(x_grids[:, i], srom_cdfs[:, i], x_grids[:, i],
                           target_cdfs[:, i], variable, y_label, plot_name_,
                           show_figure, x_limit)

    def compare_pdfs(self, variable="x", plot_dir='.',
                     plot_suffix="pdf_compare", show_figure=True,
                     save_figure=True, variable_names=None):
        """
        Generates plots comparing the srom & target pdfs for each dimension
        of the random vector.

        inputs:
            variable, str, name of variable being plotted
            plot_suffix, str, name for saving plot (will append dim & .pdf)
            plot_dir, str, name of directory to store plots
            show_figure, bool, show or not show generated plot
            save_figure, bool, save or not save generated plot
            variable_names, list of strings, names of variable in each dimension
                optional. Used for x axes labels if provided. 
        """

        x_grids = self.generate_cdf_grids()
        target_cdfs = self._target.compute_pdf(x_grids)

        (samples, probabilities) = self._SROM.get_params()

        # Start plot name string if it's being stored
        if save_figure:
            plot_name = os.path.join(plot_dir, plot_suffix)
        else:
            plot_name = None

        # Get variable names:
        if variable_names is not None:
            if len(variable_names) != self._SROM._dim:
                raise ValueError("Wrong number of variable names provided")
        else:
            variable_names = []
            for i in range(self._SROM._dim):
                if self._SROM._dim == 1:
                    variable_names.append(variable)
                else:
                    variable_names.append(variable + "_" + str(i + 1))

        if len(samples.shape) == 1:
            samples = samples.reshape((1, len(samples)))

        for i in range(self._SROM._dim):

            variable = variable_names[i]
            y_label = "f(" + variable + ")"

            # Remove latex math symbol from plot name
            if plot_name is not None:
                plot_name_ = plot_name + "_" + variable + ".pdf"
                plot_name_ = plot_name_.replace("$", "")
            else:
                plot_name_ = None

            print("samples = ", samples[:, i])
            self.plot_pdfs(samples[:, i], probabilities.flatten(),
                           x_grids[:, i], target_cdfs[:, i], variable, y_label,
                           plot_name_, show_figure)

    def compute_moment_error(self, max_moment=4):
        """
        Performs a comparison of the moments between the SROM and target, 
        calculates the percent errors up to moments of order 'max_moment'.
        Optionally generates text file with the latex source to generate
        a table.
        """

        # Get moment arrays (max_order x dim).
        srom_moments = self._SROM.compute_moments(max_moment)   
        target_moments = self._target.compute_moments(max_moment)   
        
        percent_errors = np.abs(srom_moments-target_moments)
        percent_errors = percent_errors/np.abs(target_moments)

        return percent_errors*100

    @staticmethod
    def plot_cdfs(x_grid, srom_cdf, x_target, target_cdf, x_label="x",
                  y_label="F(x)", plot_name=None, show_figure=True,
                  x_limits=None):
        """
        Plotting routine for comparing a single srom/target cdf
        """
        
        # Text formatting for plot.
        axis_font = {'fontname': 'Arial', 'size': 26, 'weight': 'normal'}
        label_font = 'Arial'
        label_size = 20
        legend_font = 22
    
        # Plot CDFs.
        fig, ax = plt.subplots(1)
        ax.plot(x_grid, srom_cdf, 'r--', linewidth=4.5, label='SROM')
        ax.plot(x_target, target_cdf, 'k-', linewidth=2.5, label='Target')
        ax.legend(loc='best', prop={'size': legend_font})
        fig.canvas.manager.set_window_title("CDF Comparison")

        # Labels/limits.
        ax.axis([min(x_grid), max(x_grid), 0, 1.1])
        if x_limits is not None:
            ax.axis([x_limits[0], x_limits[1], 0, 1.1])
        ax.set_xlabel(x_label, **axis_font)
        ax.set_ylabel(y_label, **axis_font)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname(label_font)
            label.set_fontsize(label_size)

        plt.tight_layout()

        if plot_name is not None:
            plt.savefig(plot_name)
        if show_figure:
            plt.show()    

    @staticmethod
    def plot_pdfs(samples, probabilities, x_target, target_pdf,
                  x_label="x", y_label="f(x)", plot_name=None,
                  show_figure=True):
        """
        Plotting routine for comparing a single srom/target pdf
        """
        
        # Text formatting for plot.
        axis_font = {'fontname': 'Arial', 'size': 26, 'weight': 'normal'}
        label_font = 'Arial'
        label_size = 20
        legend_font = 22
    
        # Scale SROM probabilities such that max(SROM_prob) = max(target_prob).
        scale = max(target_pdf) / max(probabilities)
        probabilities *= scale

        # Get width of bars in some intelligent way:
        x_len = max(x_target) - min(x_target)
        width = 0.1*x_len/len(samples)  # Bars take up 10% of x axis?

        # Plot PDFs
        fig, ax = plt.subplots(1)
        ax.plot(x_target, target_pdf, 'k-', linewidth=2.5, label='Target')
        ax.bar(samples, probabilities, width, color='red', label='SROM')

        ax.legend(loc='best', prop={'size': legend_font})
        fig.canvas.set_window_title("PDF Comparison")

        # Labels/limits.
        x_limits = ax.get_xlim()

        if x_limits is not None:
            ax.axis([x_limits[0], x_limits[1], 0, 1.1])

        ax.set_xlabel(x_label, **axis_font)
        ax.set_ylabel(y_label, **axis_font)

        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname(label_font)
            label.set_fontsize(label_size)

        plt.tight_layout()

        if plot_name is not None:
            plt.savefig(plot_name)
        if show_figure:
            plt.show()    

    def generate_cdf_grids(self, cdf_grid_pts=1000):
        """
        Generate numerical grids for plotting CDFs based on the 
        range of the target random vector. Return  x_grid variable with
        cdf_grid_pts along each dimension of the random vector.
        """

        x_grid = np.zeros((cdf_grid_pts, self._target._dim))

        for i in range(self._target._dim):
            grid = np.linspace(self._target.mins[i],
                               self._target.maxs[i],
                               cdf_grid_pts)
            x_grid[:, i] = grid

        return x_grid


# Need to make this whole class static
# ----------------Gross down here, don't look -----------------------
    @staticmethod
    def compare_srom_cdfs(size2srom, target, variable="x", plot_dir=".",
                          plot_suffix="CDFscompare", show_figure=True,
                          save_figure=True, variable_names=None,
                          y_limits=None, x_limits=None, x_ticks=None, 
                          cdf_y_label=False,
                          x_axis_padding=None, axis_font_size=30,
                          label_font_size=24, legend_font_size=25):
        """
        Generates plots comparing CDFs from sroms of different sizes versus 
        the target variable for each dimension of the vector.

        inputs:
            size2srom, dict, key=size of SROM (int), value = srom object
            target, TargetRV, target random variable object
            variable, str, name of variable being plotted
            plot_suffix, str, name for saving plot (will append dim & .pdf)
            plot_dir, str, name of directory to store plots
            show_figure, bool, show or not show generated plot
            save_figure, bool, save or not save generated plot
            variable_names, list of strings, names of variable in each dimension
                optional. Used for x axes labels if provided. 
            cdf_y_label, bool, use "CDF" as y-axis label? If False, uses
                            F(<variable_name>)
            x_axis_padding, int, spacing between xtick labels and x-axis
        """

        # Make x grids for plotting.
        cdf_grid_pts = 1000
        x_grids = np.zeros((cdf_grid_pts, target._dim))

        for i in range(target._dim):
            grid = np.linspace(target.mins[i],
                               target.maxs[i],
                               cdf_grid_pts)
            x_grids[:, i] = grid

        # If x_ticks is None (default), cast to list of Nones for plotting:
        if x_ticks is None:
            x_ticks = target._dim * [None]

        # Get CDFs for each size SROM.
        srom_cdfs = OrderedDict()
        for m, srom in size2srom.iteritems():
            srom_cdfs[m] = srom.compute_cdf(x_grids)

        # targetCDFs = target.compute_CDF(x_grids)
        (target_grids, targetCDFs) = target.get_plot_cdfs()

        # Start plot name string if it's being stored.
        if save_figure:
            plot_name = os.path.join(plot_dir, plot_suffix)
        else:
            plot_name = None

        # Get variable names:
        if variable_names is not None:
            if len(variable_names) != target._dim:
                raise ValueError("Wrong number of variable names provided")
        else:
            variable_names = []
            for i in range(target._dim):
                if target._dim == 1:
                    variable_names.append(variable)
                else:
                    variable_names.append(variable + "_" + str(i + 1))

        for i in range(target._dim):

            variable = variable_names[i]
            plot_name_ = plot_name + "_" + variable + ".pdf"
            if not cdf_y_label:
                y_label = r'$F($' + variable + r'$)$'
            else:
                y_label = "CDF"
   
            # Remove latex math symbol from plot name
            plot_name_ = plot_name_.translate(None, "$")

            # ---------------------------------------------
            # PLOT THIS DIMENSION:
            # Text formatting for plot
            axis_font = {'fontname': 'Arial', 'size': axis_font_size,
                         'weight': 'normal'}
            label_font = 'Arial'
            label_size = label_font_size
            legend_font = legend_font_size
       
            x_grid = x_grids[:, i]
            x_target = target_grids[:, i]
            target_cdf = targetCDFs[:, i]

            lines = ['g-', 'r:', 'b--']
            widths = [2.5, 4, 3.5]

            # Plot CDFs.
            fig, ax = plt.subplots(1)
            ax.plot(x_target, target_cdf, 'k-', linewidth=2.5, label='Target')
            for j, m in enumerate(srom_cdfs.keys()):
                label = "m = " + str(m)
                srom_cdf = srom_cdfs[m][:, i]
                ax.plot(x_grid, srom_cdf, lines[j], linewidth=widths[j],
                        label=label)

            ax.legend(loc='best', prop={'size': legend_font})

            # Labels/limits.
            ax.axis([min(x_grid), max(x_grid), 0, 1.1])

            if x_limits is not None and y_limits is None:
                ax.axis([x_limits[i][0], x_limits[i][1], 0, 1.1])

            elif x_limits is None and y_limits is not None:
                x_limits = ax.get_xlim()
                ax.axis([x_limits[0], x_limits[1], y_limits[i][0],
                         y_limits[i][1]])

            elif x_limits is not None and y_limits is not None:
                ax.axis([x_limits[i][0], x_limits[i][1],
                         y_limits[i][0], y_limits[i][1]])

            else:
                ax.axis([min(x_grid), max(x_grid), 0, 1.1])

            ax.set_xlabel(variable, **axis_font)
            ax.set_ylabel(y_label, **axis_font)

            if x_axis_padding:
                ax.tick_params(axis='x', which='major', pad=x_axis_padding)

            # Adjust tick labels:
            if x_ticks[i] is not None:
                ax.set_xticklabels(x_ticks[i])

            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontname(label_font)
                label.set_fontsize(label_size)

            plt.tight_layout()

            if plot_name_ is not None:
                plt.savefig(plot_name_)
            if show_figure:
                plt.show()

    def compare_random_variable_cdfs(self, random_variable_1, random_variable_2,
                                     variable="x", plot_dir=".",
                                     plot_suffix="CDFscompare",
                                     show_figure=True, save_figure=False,
                                     variable_names=None, x_limits=None,
                                     labels=None):
        """
        Generates plots comparing CDFs from sroms of different sizes versus 
        the target variable for each dimension of the vector.

        inputs:
            random_variable_1, SampleRandomVector, target random variable object
            random_variable_2, SampleRandomVector, target random variable object
            variable, str, name of variable being plotted
            plot_suffix, str, name for saving plot (will append dim & .pdf)
            plot_dir, str, name of directory to store plots
            show_figure, bool, show or not show generated plot
            save_figure, bool, save or not save generated plot
            variable_names, list of strings, names of variable in each dimension
                optional. Used for x axes labels if provided. 
            labels, list of str: names of random_variable_1 & random_variable_2
        """

        (random_variable_1_grids, random_variable_1_cdfs) = \
            random_variable_1.get_plot_cdfs()

        (random_variable_2_grids, random_variable_2_cdfs) = \
            random_variable_2.get_plot_cdfs()

        # Start plot name string if it's being stored.
        if save_figure:
            plot_name = os.path.join(plot_dir, plot_suffix)
        else:
            plot_name = None

        # Get variable names:
        if variable_names is not None:
            if len(variable_names) != self._target._dim:
                raise ValueError("Wrong number of variable names provided")
        else:
            variable_names = []
            for i in range(random_variable_1.dim):
                if random_variable_1.dim == 1:
                    variable_names.append(variable)
                else:
                    variable_names.append(variable + "_" + str(i + 1))

        print("variable names = ", variable_names)

        for i in range(random_variable_1.dim):

            variable = variable_names[i]
            if plot_name is not None:
                plot_name_ = plot_name + "_" + variable + ".pdf"
                # Remove latex math symbol from plot name.
                plot_name_ = plot_name_.translate(None, "$")
            else:
                plot_name_ = None

            y_label = r'$F($' + variable + r'$)$'

            # ---------------------------------------------
            # PLOT THIS DIMENSION:
            # Text formatting for plot
            axis_font = {'fontname': 'Arial', 'size': 26, 'weight': 'normal'}
            label_font = 'Arial'
            label_size = 20
            legend_font = 22
       
            x1 = random_variable_1_grids[:, i]
            cdf1 = random_variable_1_cdfs[:, i]
            x2 = random_variable_2_grids[:, i]
            cdf2 = random_variable_2_cdfs[:, i]

            # Plot CDFs.
            fig, ax = plt.subplots(1)
            if labels is None:
                labels = ["random_variable_1", "random_variable_2"]

            ax.plot(x1, cdf1, 'r--', linewidth=4.5, label=labels[0])
            ax.plot(x2, cdf2, 'b-', linewidth=2.5, label=labels[1])

            ax.legend(loc='best', prop={'size': legend_font})

            # Labels/limits.
            ax.axis([min(x1), max(x1), 0, 1.1])
            if x_limits is not None:
                ax.axis([x_limits[i][0], x_limits[i][1], 0, 1.1])
            ax.set_xlabel(variable, **axis_font)
            ax.set_ylabel(y_label, **axis_font)

            for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontname(label_font)
                label.set_fontsize(label_size)

            plt.tight_layout()

            if plot_name_ is not None:
                plt.savefig(plot_name_)
            if show_figure:
                plt.show()

    @staticmethod
    def __check_init_params(srom, target_random_vector):

        if not isinstance(target_random_vector, RandomEntity):
            raise TypeError("target_random_vector must descend from " +
                            "RandomEntity")

        if not hasattr(target_random_vector, 'compute_cdf'):
            raise TypeError("Target must define compute_cdf()")

        if not hasattr(target_random_vector, 'compute_moments'):
            raise TypeError("Target must define compute_moments()")
