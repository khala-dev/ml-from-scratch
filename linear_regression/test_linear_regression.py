import numpy as np
from linear_regression import predict, cost

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

