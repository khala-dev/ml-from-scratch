import numpy as np
from linear_regression import predict

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
    
