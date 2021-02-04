from plotting import plot_graph
from sklearn import model_selection

def partition_data(testing_amount,x_data,y_data,x_label,y_label,title):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data,y_data,test_size=testing_amount)
    plot_graph(x_train,y_train,x_label,y_label,title,x_test,y_test)