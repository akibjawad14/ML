import numpy as np
import pandas as pd
import sys
import driver
import part2
import part1
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import requests
import io
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import seaborn as sns

data_url = 'https://personal.utdallas.edu/~sxs190355/realEstateDataset.csv'
regressors = ["age", "distance_MRT", "num_stores"]
regressand = "price"
draw_plots = not (len(sys.argv) >= 2 and sys.argv[1] == 'noplot')

# retrieves dataset and performs any necessary preprocessing


def getDatasets():

    df = pd.read_csv(data_url,
                     names=["age", "distance_MRT", "num_stores", "price"])
    # print(df.head())

    mm = MinMaxScaler()
    df_transformed = pd.DataFrame(mm.fit_transform(df), columns=df.columns)
    # print(df_transformed.head())

    X = df_transformed[["age", "distance_MRT", "num_stores"]]
    Y = df_transformed["price"]

    # sns.relplot(data=df_transformed, x="age", y="price")
    # sns.relplot(data=df_transformed, x="distance_MRT", y="price")
    # sns.relplot(data=df_transformed, x="num_stores", y="price")
    # sns.boxplot(df['price'])
    plt.show()

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=5)

    return {
        'training_x_df': X_train,
        'training_y_df': y_train,
        'testing_x_df': X_test,
        'testing_y_df': y_test
    }
    # except:
    #     print('Unable to retrieve data')
    #     quit(code=1)


# presents 3d scatter plot of select dimensions, along with both models'
def plotRegressions(datasets, p1_weights_v, p2_weights_v):

    testing_x_df = datasets['testing_x_df']

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        xs=testing_x_df[:, 0],
        ys=testing_x_df[:, 1],
        zs=testing_x_df[:, 4]
    )
    plt.show()


def main():

    datasets = getDatasets()

    # Part 1 Training and Testing
    print('PART 1:Training...')
    p1_model = part1.LinRegModel(draw_plots)
    p1_training_mse, p1_weights_v = part1.train(p1_model, datasets)
    # print('The respective attribute weights are: ', np.transpose(p1_weights_v))
    p1_testing_mse = part1.test(p1_model, datasets)

    # print('\nP1 -- Testing')

    print('Test RMSE: %.5f' % (p1_testing_mse))

    # Part 2 Training and Testing
    print('\n\nP2 -- Training')
    # p2_model = linear_model.LinearRegression()
    # p2_training_mse, p2_weights_v = part2.train(p2_model, datasets)
    # p2_testing_mse = part2.test(p2_model, datasets)
    # print('P2 -- Training MSE: %.2f' % (p2_training_mse))
    # print('P2 -- Testing  MSE: %.2f' % (p2_testing_mse))
    part2.library_reg(datasets['training_x_df'], datasets['training_y_df'],
                      datasets['testing_x_df'], datasets['testing_y_df'])

    # if (draw_plots):
    #     plotRegressions(datasets, p1_weights_v, p2_weights_v)


if __name__ == "__main__":
    main()
