import time

import matplotlib.pyplot as plt


def plotScatter(y_list, x_list=None, filename="scatter_plot.png"):
    """
    Plot the list of pairs, where each pair is [xn, yn].

    Args:
    list_of_pairs: List of pairs where each pair is [xn, yn].
    """
    start_time = time.time()
    if x_list is None:

        plt.scatter([x for x in range(len(y_list))], y_list)
    else:
        plt.scatter(x_list, y_list)
    plt.xlabel("X pos")
    plt.ylabel("Y pos")
    plt.title("Scatter Plot of Pos")
    plt.grid(True)
    plt.savefig("plots/" + filename)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Plotting the {filename} took {execution_time} seconds...")
