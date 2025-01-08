
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.ticker import FuncFormatter
import cv2
from PIL import Image

# Get the path of the currently running script
current_directory = Path(__file__).parent


def make_directory(path):
    """
    This function create a directory corresponding to the argument path if it does not exist.
    Args:
        path: Path to the directory to be created.
    Returns:

    """
    if not os.path.exists(path):
        os.makedirs(path)
    print(f'The following directory exists:\n{path}')


def get_mae_factors(data, metric='MAE'):

    min_val = min(value[metric] for value in data.values())
    max_val = max(value[metric] for value in data.values())

    data_dict = dict()

    data_dict['min_val'] = (1-0.04) * min_val
    data_dict['max_val'] = (1+0.04) * max_val

    # Define ystick
    y_spaces = np.linspace(data_dict['min_val'], data_dict['max_val'], len(data) + 2)
    data_dict['y_spaces'] = y_spaces
    y_labels = [str(format(x, '.3f')) for x in y_spaces]
    y_labels[-3] = '...'
    y_labels[-2] = round(data['MSHNet']['MAE_original'], 3)
    y_labels[-1] = round((1 + 0.016) * y_labels[-2], 3)
    data_dict['y_labels'] = y_labels

    # Arrow localization
    arrow_dict = {'start_points': [(0.8, 0.04), (0.6, 0.04)],
                  'end_points': [(0.3, 0.0381), (0.38, 0.0381)],
                  'arrow_text': [0.32, 0.042],
                  }
    data_dict['arrow_dict'] = arrow_dict

    data_dict['v_min'] = 0.04
    data_dict['v_max'] = 0.07

    return data_dict

def get_f1_factors(data, metric='F1'):

    min_val = min(value[metric] for value in data.values())
    max_val = max(value[metric] for value in data.values())

    data_dict = dict()

    data_dict['min_val'] = (1 - 0.01) * min_val
    data_dict['max_val'] = (1 + 0.005) * max_val

    # Define ystick
    y_spaces = np.linspace(data_dict['min_val'], data_dict['max_val'], len(data) + 2)
    data_dict['y_spaces'] = y_spaces
    y_labels = [str(format(x, '.3f')) for x in y_spaces]
    y_labels[1] = '...'
    y_labels[2] = round(data['MSHNet']['F1_original'], 3)
    y_labels[0] = round((1 - 0.004) * y_labels[2], 3)
    data_dict['y_labels'] = y_labels

    # Arrow localization
    arrow_dict = {'start_points': [(1, 0.870), (0.65, 0.870)],
                  'end_points': [(0.34, 0.8765), (0.38, 0.877)],
                  'arrow_text': [0.4, 0.863],
                  }
    data_dict['arrow_dict'] = arrow_dict

    data_dict['v_min'] = 0.81
    data_dict['v_max'] = 0.87

    return data_dict

def scatter_plot(data, metric='MAE', plt_path=None, add_metric=False, add_arrow=True, plt_lines=False, plt_cbar=False, _plt=False):
    """
    This function visualizes the MAE or F1 Score scatter plots of the models compared
    as shown by Figure (1) of the SuperFormer paper.
    Set the metric as either MAE or F1.
    """

    if not _plt:
        return

    if metric == 'MAE':
        data_dict = get_mae_factors(data, metric=metric)
    elif metric == 'F1':
        data_dict = get_f1_factors(data, metric=metric)
    else:
        raise ValueError(f'Metric {metric} is not supported.')

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    label_fontsize = 15

    # Set labels and ticks for the first subplot
    min_val = data_dict['min_val']
    max_val = data_dict['max_val']
    ax[0].set_ylim([min_val, max_val])

    y_spaces = data_dict['y_spaces']
    y_labels = data_dict['y_labels']
    ax[0].set_yticks(y_spaces)
    ax[0].set_yticklabels(y_labels)
    ax[0].set_ylabel(metric, fontsize=label_fontsize)
    ax[0].set_xlabel('Model Params (M)', fontsize=label_fontsize)
    ax[0].set_xlim([0, 5])
    ax[0].tick_params(axis='x', labelsize=label_fontsize)
    ax[0].tick_params(axis='y', labelsize=label_fontsize)

    # Set labels and ticks for the second subplot
    ax[1].set_yticks([])
    ax[1].set_ylim([min_val, max_val])
    ax[1].set_xlabel('GFLOPs (B)', fontsize=label_fontsize)
    ax[1].set_xscale('log')
    ax[1].set_xlim([0.3, 30])
    ax[1].get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax[1].tick_params(axis='x', labelsize=15)


    # Change the border color to grey
    for a in ax:
        a.spines['top'].set_color('grey')
        a.spines['right'].set_color('grey')
        a.spines['left'].set_color('grey')
        a.spines['bottom'].set_color('grey')

    # Define color palette (single colormap based on F1 score)
    v_min, v_max = data_dict['min_val'], data_dict['max_val']
    cmap = plt.cm.Blues
    norm = plt.Normalize(vmin=v_min, vmax=v_max)

    # Create scatter plots for both axes with F1-based color intensity
    color_palette_ = []
    w = 1.
    for i, (net, value) in enumerate(data.items()):

        # color = color_palette[i]
        color = cmap(w * norm(value[metric]))
        color_palette_.append(color)
        size = value['size']

        ax[0].scatter(value['Params'], value[metric], s=size, marker='o', color=color)
        ax[1].scatter(value['FLOPs'], value[metric], s=size, marker='o', color=color)

        # Highlight specific models with a red outline
        if net in ["Ours (XS)", "Ours (S)"]:
            ax[0].scatter(value['Params'], value[metric], s=size * 1.5, facecolors='none', edgecolors='red', linewidth=2)
            ax[1].scatter(value['FLOPs'], value[metric], s=size * 1.5, facecolors='none', edgecolors='red', linewidth=2)
        # else:
        #     ax[0].scatter(value['Params'], value['MAE'], s=size * 1.5, facecolors='none', edgecolors='green', linewidth=2)
        #     ax[1].scatter(value['FLOPs'], value['MAE'], s=size * 1.5, facecolors='none', edgecolors='green', linewidth=2)

    # Create custom legend handles with sizes based on sorted data
    sorted_handles = [
        plt.Line2D(
            [0], [0], marker='o', color='w', label=net,
            markersize=np.sqrt(value['size']),
            markerfacecolor=color_palette_[i],
            markeredgewidth=2 if net in ["Ours (XS)", "Ours (S)"] else 0,
            markeredgecolor='red' if net in ["Ours (XS)", "Ours (S)"] else 'none'
        )
        for i, (net, value) in enumerate(data.items())
    ]

    # Add the legend to ax[1]
    ax[1].legend(handles=sorted_handles, loc='upper left', bbox_to_anchor=(1, 1), fontsize=15, labelspacing=1.2)

    # Create colorbar for score intensity on the left side of the left window (ax[0])
    if plt_cbar:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax[0], orientation='vertical', location='left', fraction=0.05, pad=0.15)
        cbar.set_label(metric, fontsize=label_fontsize)
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.yaxis.set_label_position('left')
        cbar.ax.yaxis.tick_left()


    if plt_lines:
        for i, (net, value) in enumerate(data.items()):
            # Horizontal and vertical lines for each circle in ax[0]
            ax[0].axhline(y=value[metric], color=color_palette_[i], linestyle='--', linewidth=0.8)
            ax[0].axvline(x=value['Params'], color=color_palette_[i], linestyle='--', linewidth=0.8)

            # Horizontal and vertical lines for each circle in ax[1]
            ax[1].axhline(y=value[metric], color=color_palette_[i], linestyle='--', linewidth=0.8)
            ax[1].axvline(x=value['FLOPs'], color=color_palette_[i], linestyle='--', linewidth=0.8)

            if net == "MEANet":
                print(f"Skipping {net}")
                continue

    if add_metric:
        ax[0].text(
            0, value[metric], f'{value[metric]:.3f}',
            ha='right', va='center', fontsize=15, color='black')

    if add_arrow:
        arrow_dict = data_dict['arrow_dict']
        for i in range(2):
            ax[i].annotate('', xy=arrow_dict['end_points'][i], xytext=arrow_dict['start_points'][i],
                           arrowprops=dict(facecolor='red', edgecolor='red', arrowstyle='->', lw=2))
            ax[i].text(arrow_dict['arrow_text'][0],
                       arrow_dict['arrow_text'][1],
                       "Enhanced\nEfficiency!", color='red', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'{plt_path}/scatter_{metric}.pdf', format='pdf')
    plt.show()
    print('Well-Done.')

def sf_recos_err(recon_err_sp, recon_err_ds, _plt=False):

    error_sp = np.load(recon_err_sp)
    error_ds = np.load(recon_err_ds)

    # Define the range for both histograms based on the minimum and maximum values across both arrays
    min_value = min(error_sp.min(), error_ds.min())
    max_value = max(error_sp.max(), error_ds.max())

    # Set up bins to ensure consistent bin width across both histograms
    num_bins = 150
    bin_edges = np.linspace(min_value, max_value,
                            num_bins + 1)  # +1 because edges include both start and end of each bin

    plt.figure(figsize=(10, 6))
    plt.hist(error_sp, bins=bin_edges, alpha=0.5, label='SuperFormer', color='blue')
    plt.hist(error_ds, bins=bin_edges, alpha=0.5, label='Downsample', color='orange')
    plt.xlabel('MAE', fontsize=20)
    plt.ylabel('Frequency Count', fontsize=20)

    # Set font size for x and y tick labels
    plt.tick_params(axis='x', labelsize=18)
    plt.tick_params(axis='y', labelsize=18)

    # Set legend and its font size
    plt.rcParams['legend.fontsize'] = 20

    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{current_directory}/reconstruction_errors.pdf')
    plt.show()

    print('Well-Done.')


def resample_2d(points, N):

    xc = points[:, 0].tolist() + [points[0, 0]]
    yc = points[:, 1].tolist() + [points[0, 1]]

    dx = np.diff(xc)
    dy = np.diff(yc)

    dS = np.sqrt(dx ** 2 + dy ** 2)
    dS = np.array([0] + dS.tolist())

    d = np.cumsum(dS)

    perim = d[-1]

    ds = perim / N
    dSi = ds * np.arange(0, N)
    dSi[-1] = dSi[-1] - 0.005

    xi = np.interp(dSi, d, xc)
    yi = np.interp(dSi, d, yc)
    return xi, yi

def euc_distance(pt1, pt2):
    y_diff = np.abs(pt2[1] - pt1[1])
    x_diff = np.abs(pt2[0] - pt1[0])
    return np.sqrt(y_diff ** 2 + x_diff ** 2)


def plot_contours(fig, contour_array, xi, yi, xi_t, yi_t, ax, font_sizes, i_x=0, i_y=0, title='Original'):

    # Plot the original contours as purple dashed lines
    line1, = ax[i_x, i_y].plot(xi, yi, '--', color='purple', label='Original')

    # Plot the transformed contours as solid black lines
    line2, = ax[i_x, i_y].plot(xi_t, yi_t, '-', color='black', label='Transformed')

    # Scatter plot for the first and second points with specific colors for the original contours
    scatter1 = ax[i_x, i_y].scatter(xi[0], yi[0], c='blue', label='1st', s=50, edgecolors='blue', zorder=5)
    scatter2 = ax[i_x, i_y].scatter(xi[1], yi[1], c='green', label='2nd', s=50, edgecolors='green', zorder=5)

    # Adding legend manually to control the order and label
    if i_x == 0 and i_y == 0:
        # ax[i_x, i_y].set_ylabel('Contours', fontsize=font_sizes['ylabel_fontsize'])
        ax[i_x, i_y].legend(handles=[line1, line2, scatter1, scatter2],
                            labels=['Original', 'Transformed', 'Start', 'Second'],
                            loc='upper left', bbox_to_anchor=(-1.1, 1), fontsize=font_sizes['tick_fontsize'],
                            title='Contours')

        # fig.legend(handles=[line1, line2, scatter1, scatter2],
        #            labels=['Original', 'Transformed', '1st', '2nd'],
        #            loc='upper center', bbox_to_anchor=(0.5, 1.), ncol=4, fontsize=font_sizes['tick_fontsize'])

    # Set the title and axis labels with custom font sizes
    ax[i_x, i_y].set_title(title, fontsize=font_sizes['title_fontsize'])
    # ax[i_x, i_y].set_xlabel('X-axis', fontsize=font_sizes['xlabel_fontsize'])

    # Set the font size for axis ticks
    ax[i_x, i_y].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])



def get_fft(contour_array):

    contour_complex = np.empty(contour_array.shape[:-1], dtype=complex)
    contour_complex.real = contour_array[:, 0]
    contour_complex.imag = contour_array[:, 1]
    fourier_result = np.fft.fft(contour_complex)

    return fourier_result



def sf_FT(data_file, plt_path=None, _plt=False):
    """
    This function plots the Fourier Transform results of the SuperFormer Figure (S.4) of the paper.
    Args:
        data_file:
        plt_path:
        _plt:

    Returns:

    """

    if not _plt:
        return

    data = np.load(data_file)
    rectangle = (data * 255).astype(np.uint8)

    contour, hierarchy = cv2.findContours(rectangle, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points = contour[0][:, 0, :]
    # ax[0].scatter(points[:,0], points[:, 1])

    distances = []
    for i in range(len(points)):
        if i == len(points) - 1:
            distances.append(euc_distance(points[i], points[0]))
        else:
            distances.append(euc_distance(points[i], points[i + 1]))

    # plt.plot(distances)
    # plt.show()

    N = 70
    # ORIGINAL: Create Contours
    xi, yi = resample_2d(points, N)
    contour_array_orig = np.stack((xi, yi), axis=1)
    fft_orig = get_fft(contour_array_orig)
    phase_orig = np.arctan2(fft_orig.imag, fft_orig.real)

    # TRANSLATION
    tr_factor = 30
    contour_array_tr = np.stack((xi, yi), axis=1) + tr_factor
    fft_tr = get_fft(contour_array_tr)
    phase_tr = np.arctan2(fft_tr.imag, fft_tr.real)

    # SCALING
    sc_factor = 3
    contour_array_sc = np.stack((xi * sc_factor, yi * sc_factor), axis=1)
    fft_sc = get_fft(contour_array_sc)
    phase_sc = np.arctan2(fft_sc.imag, fft_sc.real)

    # ROTATION
    rot_angle = 30
    im = Image.fromarray(rectangle)
    rotated = im.rotate(rot_angle)
    rectangle_rotated = np.array(rotated)
    contour, hierarchy = cv2.findContours(rectangle_rotated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    points_rotated = contour[0][:, 0, :]
    xi_rot, yi_rot = resample_2d(points_rotated, N)
    contour_array_rot = np.stack((xi_rot, yi_rot), axis=1)
    fft_rot = get_fft(contour_array_rot)
    phase_rot = np.arctan2(fft_rot.imag, fft_rot.real)
    print(fft_rot[-1])
    print(fft_orig[-1])

    # Calculate the differences between each of the transformed values and the original
    amp_diff_tr = abs(fft_tr) - abs(fft_orig)
    amp_diff_sc = abs(fft_sc) - abs(fft_orig)
    amp_diff_rot = abs(fft_rot) - abs(fft_orig)

    diff_phase_tr = phase_tr - phase_orig
    diff_phase_sc = phase_sc - phase_orig
    diff_phase_rot = phase_rot - phase_orig

    # --------------------- PLOT Figure --------------------------------
    fig, ax = plt.subplots(5, 3, figsize=(10, 10))
    font_sizes = {'title_fontsize': 12,
                  'ylabel_fontsize': 10,
                  'xlabel_fontsize': 10,
                  'tick_fontsize': 12,}

    # CONTOURS
    plot_contours(fig, contour_array_tr, xi, yi, xi+tr_factor, yi+tr_factor, ax, font_sizes,
                  i_x=0, i_y=0, title=f'Translated by {tr_factor}')
    plot_contours(fig, contour_array_sc, xi, yi, xi*sc_factor, yi*sc_factor, ax, font_sizes,
                  i_x=0, i_y=1, title=f'Scaled by {sc_factor}')
    plot_contours(fig, contour_array_rot, xi, yi, xi_rot, yi_rot, ax, font_sizes,
                  i_x=0, i_y=2, title=f'Rotated {rot_angle} degrees')

    # AMPLITUDE
    ax[1, 0].stem(np.linspace(0, np.pi, len(fft_tr))[1:], abs(fft_tr[1:]), 'k', markerfmt=" ", basefmt="-k")
    ax[1, 1].stem(np.linspace(0, np.pi, len(fft_sc))[1:], abs(fft_sc[1:]), 'k', markerfmt=" ", basefmt="-k")
    ax[1, 2].stem(np.linspace(0, np.pi, len(fft_rot))[1:], abs(fft_rot[1:]), 'k', markerfmt=" ", basefmt="-k")

    ax[1, 0].legend(labels=['Amplitude (fs)'], loc='upper left', bbox_to_anchor=(-1.1, 0.7), fontsize=11)
    ax[1, 0].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])
    ax[1, 1].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])
    ax[1, 2].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])


    # AMPLITUDE Differences
    ax[2, 0].stem(np.linspace(0, np.pi, len(amp_diff_tr))[1:], amp_diff_tr[1:], 'r', markerfmt=" ", basefmt="-r")
    ax[2, 1].stem(np.linspace(0, np.pi, len(amp_diff_sc))[1:], amp_diff_sc[1:], 'r', markerfmt=" ", basefmt="-r")
    ax[2, 2].stem(np.linspace(0, np.pi, len(amp_diff_rot))[1:], amp_diff_rot[1:], 'r', markerfmt=" ",  basefmt="-r")

    ax[2, 0].legend(labels=['Amplitude\nDifference'], loc='upper left', bbox_to_anchor=(-1.1, 0.64), fontsize=font_sizes['tick_fontsize'])
    ax[2, 0].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])
    ax[2, 1].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])
    ax[2, 2].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])

    # PHASE
    ax[3, 0].stem(np.linspace(0, np.pi, len(phase_tr))[1:], phase_tr[1:], 'k', markerfmt=" ", basefmt="-k")
    ax[3, 1].stem(np.linspace(0, np.pi, len(phase_sc))[1:], phase_sc[1:], 'k', markerfmt=" ", basefmt="-k")
    ax[3, 2].stem(np.linspace(0, np.pi, len(phase_rot))[1:], phase_rot[1:], 'k', markerfmt=" ", basefmt="-k")

    ax[3, 0].legend(labels=['Phase (fs)'], loc='upper left', bbox_to_anchor=(-1.1, 0.65), fontsize=font_sizes['tick_fontsize'])
    ax[3, 0].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])
    ax[3, 1].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])
    ax[3, 2].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])

    # PHASE Differences
    ax[4, 0].stem(np.linspace(0, np.pi, len(diff_phase_tr))[1:], diff_phase_tr[1:], 'r', markerfmt=" ", basefmt="-r")
    ax[4, 1].stem(np.linspace(0, np.pi, len(diff_phase_sc))[1:], diff_phase_sc[1:], 'r', markerfmt=" ", basefmt="-r")
    ax[4, 2].stem(np.linspace(0, np.pi, len(diff_phase_rot))[1:], diff_phase_rot[1:], 'r', markerfmt=" ",  basefmt="-r")

    ax[4, 0].legend(labels=['Phase\nDifference'], loc='upper left', bbox_to_anchor=(-1.1, 0.65), fontsize=font_sizes['tick_fontsize'])
    ax[4, 0].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])
    ax[4, 1].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])
    ax[4, 2].tick_params(axis='both', labelsize=font_sizes['tick_fontsize'])

    plt.tight_layout()
    plt.savefig(f'{plt_path}/ft_vis.pdf')
    plt.show()

    print('Well-Done.')


def data_sf():
    """ Get the results values."""
    data = {
        'Ours (XS)': {'Params': 1.12, 'FLOPs': 0.46, 'MAE': 0.0530, 'F1': 0.8470, 'size': 100, },
        'HVPNet':    {'Params': 1.23, 'FLOPs': 1.10, 'MAE': 0.0580, 'F1': 0.8390, 'size': 150, },
        'SAMNet':    {'Params': 1.33, 'FLOPs': 0.50, 'MAE': 0.0580, 'F1': 0.8350, 'size': 200, },
        'Ours (S)':  {'Params': 2.18, 'FLOPs': 1.63, 'MAE': 0.0390, 'F1': 0.8760, 'size': 250, },
        'SeaNet':    {'Params': 2.76, 'FLOPs': 1.70, 'MAE': 0.0450, 'F1': 0.8540, 'size': 300, },
        'MEANet':    {'Params': 3.27, 'FLOPs': 9.62, 'MAE': 0.0454, 'F1': 0.8630, 'size': 400, },
        'ISAANet':   {'Params': 3.59, 'FLOPs': 4.34, 'MAE': 0.0430, 'F1': 0.8750, 'size': 430, },
        'MSHNet':    {'Params': 4.07, 'FLOPs': 6.11, 'MAE': 0.0651, 'F1': 0.8221, 'size': 500, 'MAE_original': 0.1251, 'F1_original': 0.7046, },
        'CorrNet':   {'Params': 4.09, 'FLOPs': 21.1, 'MAE': 0.0466, 'F1': 0.8470, 'size': 500},
    }
    return data


if __name__ == '__main__':


    plt_path = os.path.join(current_directory, 'figs')
    make_directory(plt_path)

    data = data_sf()
    scatter_plot(data, metric='MAE', plt_path=plt_path, add_metric=False, add_arrow=True, _plt=True)

    data_file = f'{current_directory}/sample_sp.npy'
    sf_FT(data_file, plt_path, _plt=True)


    recon_err_sp = f'{current_directory}/reconstruction_error_sp.npy'
    recon_err_ds = f'{current_directory}/reconstruction_error_ds.npy'
    sf_recos_err(recon_err_sp, recon_err_ds, _plt=True)
