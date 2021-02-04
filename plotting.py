import matplotlib.pyplot as plt

def plot_graph(x_data, y_data, x_label = "", y_label = "", graph_title = "", x1_data = [], y1_data = []):
    plt.scatter(x_data,y_data)
    if len(x1_data) and len(y1_data) != 0:
        plt.scatter(x1_data,y1_data)
    plt.xlabel(x_label), plt.ylabel(y_label), plt.title(graph_title)
    plt.show()
    plt.close()