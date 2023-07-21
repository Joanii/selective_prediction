import matplotlib
import matplotlib.pyplot as plt
import os


def plot_risk_over_coverage(risks, coverages, attack_name, epsilon, folder_path):
    markers = marker_config()
    colors = color_code()
    update_plotspecs()
    plt.figure()
    for key, values in risks.items():
        plt.plot(coverages[key], values, linestyle='None', marker=markers[key][0], label=key,
                 markersize=markers[key][1], color=colors[key])
    plt.ylabel('risk')
    plt.ylim(0)
    plt.xlabel('coverage')
    plt.subplots_adjust(bottom=0.15, top=0.9, left=0.15, right=0.98)
    plt.legend(ncol=len(risks), loc='upper center', bbox_to_anchor=(0.47, 1.15), handletextpad=0.0001,
               columnspacing=0.05, frameon=False)
    save_path = os.path.join(folder_path, f'{attack_name}_{epsilon}')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    plt.savefig(f'{save_path}.png')
    plt.savefig(f'{save_path}.eps')
    plt.show()


def update_plotspecs():
    nice_fonts = {
                'font.family': 'serif',
                'axes.labelsize': 10,
                'font.size': 10,
                'legend.fontsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
    }
    matplotlib.rcParams.update(nice_fonts)


def marker_config():
    marker_size = 6  # 7
    return {
        'VAE': ('o', marker_size),
        'Softmax': ('H', marker_size),
        'Joint': ('*', marker_size+2),
        'Sel': ('d', marker_size),
        'Joint-S': ('p', marker_size),
        'Latent': ('H', marker_size),
    }


def color_code():
    return {
        'VAE': 'tab:blue',
        'Softmax': 'tab:orange',
        'Joint': 'black',
        'Sel': 'tab:green',
        'Joint-S': 'dimgrey',
        'Latent': 'darkgrey'
    }
