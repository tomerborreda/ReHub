import socket
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import torch



### Memory Snapshot Utils ###


TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


def start_record_memory_history() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       return

   print("Starting snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(
       max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
   )


def stop_record_memory_history() -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not recording memory history")
       return

   print("Stopping snapshot record_memory_history")
   torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot(prefix: str = "") -> None:
   if not torch.cuda.is_available():
       print("CUDA unavailable. Not exporting memory snapshot")
       return

   # Prefix for file names.
   host_name = socket.gethostname()
   timestamp = datetime.now().strftime(TIME_FORMAT_STR)
   file_prefix = f"pickles/{prefix}_{host_name}_{timestamp}"

   try:
       print(f"Saving snapshot to local file: {file_prefix}.pickle")
       torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
   except Exception as e:
       print(f"Failed to capture memory snapshot {e}")
       return




### Utilization and Bhattacharyya Metrics Utils ###


def set_style():
    # Reset to default style to ensure white background
    plt.style.use('default')

    # Set global font properties for consistency
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'figure.dpi': 300,  
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.0,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'figure.facecolor': 'white', 
        'axes.facecolor': 'white',
    })

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)

    return ax


def save_bar_graph(data, title, xlabel, ylabel, filename):
    ax = set_style()
    ax.bar(data[0], data[1], edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'./metrics/{filename}.pdf', format='pdf')
    plt.close()


def save_cumulative_graph(data, title, xlabel, ylabel, filename):
    ax = set_style()
    for i, yi in enumerate(data[1]):
        ax.plot(data[0], yi, label=f'Layer {i+1}, AUC={np.trapz(yi/100, data[0], dx=10):.2f}')

    plt.legend(loc='lower right')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(f'./metrics/{filename}.pdf', format='pdf')
    plt.close()


def generate_utilization_graph(utilization, total_num_graphs, split):
    xs = [f'{x}%-{x+10}%' for x in range(0, 91, 10)]

    utilization = np.array(utilization)
    ys = [np.sum((utilization >= 0) & (utilization <= 10/100))] + [np.sum((utilization > (x*10)/100) & (utilization <= ((x+1)*10)/100)) for x in range(1, 10)]
    
    assert np.sum(ys) == total_num_graphs
    ys = (np.array(ys)/total_num_graphs) * 100
    
    save_bar_graph((xs, ys), f'Utilization Distribution ({split})', 'Utilization %', '% of Graphs', f'utilization_{split}')


def generate_bhattacharyya_graph(bhattacharyya, total_num_graphs, split):
    xs = [f'{x}%-{x+10}%' for x in range(0, 91, 10)]

    bhattacharyya = np.array(bhattacharyya)
    ys = [np.sum((bhattacharyya >= 0) & (bhattacharyya <= 10/100))] + [np.sum((bhattacharyya > (x*10)/100) & (bhattacharyya <= ((x+1)*10)/100)) for x in range(1, 10)]
    
    assert np.sum(ys) == total_num_graphs
    ys = (np.array(ys)/total_num_graphs) * 100
    
    save_bar_graph((xs, ys), f'Bhattacharyya Distribution ({split})', 'Bhattacharyya %', '% of Graphs', f'bhattacharyya_{split}')


def generate_utilization_cumulative_graph(utilization, total_num_graphs, split):
    xs = list(range(0, 101))
    
    y = []
    for u in utilization:
        u = np.array(u)
        ys = [np.sum(u <= x/100) for x in xs]
        ys = np.array(ys)/total_num_graphs
        y.append(ys * 100)
    
    save_cumulative_graph((xs, y), f'Hub-Island Distribution ({split})', 'Hub-Island %', 'Cumulative % of Graphs', f'utilization_cumulative_{split}')


def generate_bhattacharyya_cumulative_graph(bhattacharyya, total_num_graphs, split):
    xs = list(range(0, 101))
    
    y = []
    for u in bhattacharyya:
        u = np.array(u)
        ys = [np.sum(u <= x/100) for x in xs]
        ys = np.array(ys)/total_num_graphs
        y.append(ys * 100)
    
    save_cumulative_graph((xs, y), f'Bhattacharyya Distribution ({split})', 'Bhattacharyya %', 'Cumulative % of Graphs', f'bhattacharyya_cumulative_{split}')
