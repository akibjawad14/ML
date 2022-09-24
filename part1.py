import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import random
import math
from sklearn.metrics import mean_squared_error, r2_score


class LinRegModel(object):

    def __init__(self, draw_plots):
        self.draw_plots = draw_plots

    def drawPlots(self, training_log, descents):

        # 3d gradient descent demonstration
        if (len(self.regressands) > 2):     # ensure enough descents and dimensions to plot
            fig1 = plt.figure(1)
            ax_a1 = plt.axes(projection='3d')
            # selects two regressands whose weight to plot
            regressand_indices = (2, 3)
            ax_a1.set_xlabel(
                self.regressands[regressand_indices[0]] + ' weight')
            ax_a1.set_ylabel(
                self.regressands[regressand_indices[1]] + ' weight')
            ax_a1.set_zlabel('MSE')
            ax_a1.set_title('Gradient Descent Visualization')

            def update(val):
                ax_a1.clear()
                regressand_indices = (int(slider1.val), int(slider2.val))
                ax_a1.set_xlabel(
                    self.regressands[regressand_indices[0]] + ' weight')
                ax_a1.set_ylabel(
                    self.regressands[regressand_indices[1]] + ' weight')
                ax_a1.set_zlabel('MSE')
                weight_a_v = training_log[:, 5+regressand_indices[0]]
                weight_b_v = training_log[:, 5+regressand_indices[1]]
                mse_col_v = training_log[:, 2]
                descent_v = training_log[:, 0]
                step_v = training_log[:, 1]
                if (descents == 1):
                    ax_a1.scatter3D(
                        xs=weight_a_v,  # weight
                        ys=weight_b_v,  # weight
                        zs=mse_col_v,   # MSE
                        c=step_v,       # step
                        cmap='plasma'
                    )
                    ax_a1.plot3D(xs=weight_a_v, ys=weight_b_v,
                                 zs=mse_col_v)  # connect the dots
                else:
                    ax_a1.scatter3D(
                        xs=weight_a_v,  # weight
                        ys=weight_b_v,  # weight
                        zs=mse_col_v,   # MSE
                        c=descent_v,    # descent or step
                        cmap='Set1'
                    )
                fig1.canvas.draw()

            axcolor = 'lightgoldenrodyellow'
            ax_regressand1 = plt.axes(
                [0.1, 0.25, 0.1, 0.025], facecolor=axcolor)
            ax_regressand2 = plt.axes(
                [0.1, 0.2, 0.1, 0.025], facecolor=axcolor)
            slider1 = Slider(ax_regressand1, 'Regressand 1', 0, len(
                self.regressands)-1, valinit=regressand_indices[0], valstep=1)
            slider2 = Slider(ax_regressand2, 'Regressand 2', 0, len(
                self.regressands)-1, valinit=regressand_indices[1], valstep=1)
            update(1)
            slider1.on_changed(update)
            slider2.on_changed(update)

        # plot of MSE, all weights, and learning rate, over steps for a single descent
        if (descents == 1):
            fig2, (ax_b1, ax_b2, ax_b3) = plt.subplots(
                3, constrained_layout=True)
            for idx, regressand in enumerate(self.regressands):
                color = (random.random(), random.random(), random.random())
                ax_b2.plot(
                    training_log[:, 1], training_log[:, 5+idx], c=color, label=regressand+' weight')
            ax_b2.plot(training_log[:, 1], training_log[:, 4],
                       c=color, label='bias')   # plot bias
            ax_b2.legend(fontsize='small', bbox_to_anchor=(
                1.01, 1), loc="upper left")
            ax_b2.set_ylabel('Weight')
            ax_b1.set_ylabel('MSE')
            ax_b1.plot(training_log[:, 1],
                       training_log[:, 2], 'b-', label='MSE')
            ax_b1.set_title(
                'Step-based MSE, Weight, and Learning Rate Analysis')
            ax_b3.plot(training_log[:, 1], training_log[:,
                       3], 'b-', label='Learning Rate')
            ax_b3.set_ylabel('Learning Rate')
            ax_b3.set_xlabel('Step')

        plt.show()

    def calcMSE(self, err_v):
        mse_m = err_v.transpose().dot(err_v) / (2 * len(err_v))
        return (np.sqrt(mse_m[0, 0]))

    def calcErrV(self, x_m, w_v, y_v):
        h_v = x_m.dot(w_v)
        err_v = h_v - y_v
        return err_v

    def calcR2(self, x_m, final_weights, y_v):
        h_v = x_m.dot(final_weights)
        r2 = r2_score(y_v, h_v)
        print('Training R2: ', round(r2, 5))
        return r2

    def getNewWeight(self, old_weight, learning_rate, err_v, xi_v):
        sum = 0
        # loop over all measurements assoc w/ specific parameter
        for j, xi in enumerate(xi_v):
            # sum that point's error times the point's dimension corresponding to weight
            sum += err_v[j].item() * xi
        gradient = sum / len(xi_v)
        new_weight = old_weight - learning_rate * gradient
        return new_weight

    def train(self, training_x_df, training_y_df, descents=1, learning_rate=2.03, delta_weight_threshold=0.0001):
        self.regressands = list(training_x_df)
        self.regressor = list(training_y_df)[0]
        # vector of true area values
        true_v = training_y_df.to_numpy().reshape(-1, 1)
        data_m = training_x_df.to_numpy()
        data_m = np.hstack((np.ones((len(data_m), 1)), data_m)
                           )          # matrix of data points
        # initialize log, later used for graphs and analysis
        training_log = np.zeros((1, data_m.shape[1] + 4))
        best_MSE = math.inf

        for descent in range(descents):

            # randomly initialize weights vector
            weights_v = (np.random.random_sample(
                (len(data_m[0]), 1)) - (1/2)) / 5
            # calculate error vector
            err_v = self.calcErrV(x_m=data_m, w_v=weights_v, y_v=true_v)
            # training_log = np.vstack((training_log, np.hstack(( weights_v.T, np.array([self.calcMSE(err_v), 0]).reshape(1,2) ))))

            step = 1
            while (step < 50000):

                old_weights_v = np.array(weights_v, copy=True)
                old_err_v = err_v
                old_MSE = self.calcMSE(err_v)

                # zeroed vector of weights
                new_weights_v = np.zeros((len(weights_v), 1))
                for i, xi_v in enumerate(data_m.transpose()):   # adjust each weight
                    new_weights_v[i] = self.getNewWeight(
                        old_weight=weights_v[i], learning_rate=learning_rate, err_v=err_v, xi_v=xi_v)
                weights_v = new_weights_v
                err_v = self.calcErrV(x_m=data_m, w_v=weights_v, y_v=true_v)
                new_MSE = self.calcMSE(err_v)

                delta_weights_v = np.absolute(old_weights_v - new_weights_v)
                if ((delta_weights_v < delta_weight_threshold).all()):  # end condition
                    iter_MSE = new_MSE
                    break
                # if gradient ascent by overstepping
                elif (new_MSE > old_MSE):
                    # throttle learning rate
                    learning_rate = learning_rate * 0.99
                    weights_v = old_weights_v                                       # revert weights
                    err_v = self.calcErrV(
                        x_m=data_m, w_v=weights_v, y_v=true_v)    # revert err_v
                else:
                    # log more of earlier steps
                    if (self.draw_plots and step % math.ceil(math.log(step+2, 1.01)/2) == 0):
                        training_log = np.vstack((training_log, np.hstack(
                            (np.array([descent, step, new_MSE, learning_rate]).reshape(1, 4), weights_v.T))))
                    if (step % 100 == 0):
                        print(
                            f'Step {step} \t MSE {new_MSE}')  # \t Weights {weights_v}')
                    # accelerate learning rate
                    learning_rate = learning_rate * 1.002
                    step += 1

            if (new_MSE < best_MSE):
                best_MSE = new_MSE
                self.weights_v = weights_v

        # After all descents
        # training log post processing
        if (training_log.shape[0] != 1):
            # remove zeros row from training_log
            training_log = np.delete(training_log, obj=0, axis=0)

        if (self.draw_plots):
            self.drawPlots(training_log, descents)

        print('\nP1 -- Training')
        print('The respective attribute weights are: ',
              np.delete(np.transpose(self.weights_v), 0))
        print('The intercept is: ',
              self.weights_v[1])
        self.calcR2(data_m, self.weights_v, true_v)
        print('Training RMSE: %.5f' % (best_MSE))

        return best_MSE

    def test(self, testing_x_df, testing_y_df):
        # vector of true area values
        true_v = testing_y_df.to_numpy().reshape(-1, 1)
        data_m = testing_x_df.to_numpy()
        data_m = np.hstack((np.ones((len(data_m), 1)), data_m)
                           )          # matrix of data points
        predicted_v = data_m.dot(self.weights_v).reshape(-1, 1)
        mse = self.calcMSE(predicted_v - true_v)
        return predicted_v


def train(model, datasets):
    mse = model.train(
        training_x_df=datasets['training_x_df'],
        training_y_df=datasets['training_y_df'],
        # training hyperparameters
        descents=1, learning_rate=0.006, delta_weight_threshold=0.0001
    )
    return mse, model.weights_v


def test(model, datasets):
    predictions_v = model.test(
        testing_x_df=datasets['testing_x_df'],
        testing_y_df=datasets['testing_y_df']
    )

    # predictions_v )
    print('\nP1 -- Testing')

    print('Test R2: ', round(
        r2_score(datasets['testing_y_df'], predictions_v), 5))
    return model.calcMSE(predictions_v - datasets['testing_y_df'].to_numpy())

# r2 = r2_score(y_train, y_train_predict) #predictions_v
