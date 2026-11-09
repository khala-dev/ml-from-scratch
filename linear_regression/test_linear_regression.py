import numpy as np
from linear_regression import predict, cost, gradient_descent

#test cases for predict function
def test_predict_matches_hand_calculated():
    # Line y_hat = 6.5 + 1.8x
    x = np.array([-1, 0, 1, 2, 5])
    expected = np.array([4.7, 6.5, 8.3, 10.1, 15.5])
    assert np.allclose(predict(6.5, 1.8, x), expected)

def test_intercept_is_prediction_at_zero():
    # when x = 0, the prediction equals theta0
    assert predict(6.5, 1.8, 0) == 6.5
    
def test_slope_is_change_pre_unit_of_x():
    #increasing x by 1 increases the prediction by theta1
    assert np.isclose(predict(6.5, 1.8, 3) - predict(6.5, 1.8, 2), 1.8)


#test cases for cost function
def test_cost_is_zero_for_perfect_fit():
    # when the model perfectly fits the data, cost is zero
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3]) # y = 0 + 1x
    assert cost(0, 1, x, y) == 0

def test_cost_matches_hand_calculation():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    assert np.isclose(cost(0, 0.5, x, y), 3.5 / 6)
    assert np.isclose(cost(0, 0, x, y), 14 / 6)

def test_cost_grows_as_line_moves_away():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    assert cost(0, 1, x, y) < cost(0, 0.5, x, y) < cost(0, 0, x, y)


#test cases for gradient descent function
def test_gradient_descent_one_step_matches_hand_calculation():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    theta0, theta1, _ = gradient_descent(x, y, alpha=0.1, n_iterations=1)
    assert np.isclose(theta0, 0.2)
    assert np.isclose(theta1, 1.4 / 3)


def test_gradient_descent_finds_the_perfect_line():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    theta0, theta1, _ = gradient_descent(x, y, alpha=0.1, n_iterations=2000)
    assert np.isclose(theta0, 0, atol=1e-3)
    assert np.isclose(theta1, 1, atol=1e-3)


def test_cost_decreases_with_small_alpha():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    _, _, history = gradient_descent(x, y, alpha=0.1, n_iterations=50)
    assert all(a >= b for a, b in zip(history, history[1:]))


def test_cost_grows_when_alpha_is_too_large():
    x = np.array([1, 2, 3])
    y = np.array([1, 2, 3])
    _, _, history = gradient_descent(x, y, alpha=1, n_iterations=10)
    assert history[-1] > history[0]
    


AREA = np.array([50, 65, 80, 95, 110])
PRICE = np.array([900, 1100, 1500, 1600, 2000])


def test_runs_all_iterations_when_no_tolerance():
    _, _, history = gradient_descent(AREA, PRICE, alpha=0.0001, n_iterations=50)
    assert len(history) == 50


def test_stops_early_when_cost_change_is_below_tolerance():
    _, _, history = gradient_descent(
        AREA, PRICE, alpha=0.0001, n_iterations=100000, tolerance=0.01
    )
    assert len(history) == 9


def test_cost_is_close_to_minimum_when_stopped_early():
    _, _, history = gradient_descent(
        AREA, PRICE, alpha=0.0001, n_iterations=100000, tolerance=0.01
    )
    assert history[-1] < 1920


def test_cost_grows_on_housing_data_when_alpha_is_too_large():
    _, _, history = gradient_descent(AREA, PRICE, alpha=0.0005, n_iterations=10)
    assert history[-1] > history[0]