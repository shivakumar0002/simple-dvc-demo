import pytest
from prediction_service.prediction import form_response, api_response
import prediction_service

input_data = {
    "incorrect_range": {
        "fixed_acidity": 7897897, 
        "volatile_acidity": 555, 
        "citric_acid": 99, 
        "residual_sugar": 99, 
        "chlorides": 12, 
        "free_sulfur_dioxide": 789, 
        "total_sulfur_dioxide": 75, 
        "density": 2, 
        "pH": 33, 
        "sulphates": 9, 
        "alcohol": 9
    },

    "correct_range": {
        "fixed_acidity": 5, 
        "volatile_acidity": 1, 
        "citric_acid": 0.5, 
        "residual_sugar": 10, 
        "chlorides": 0.5, 
        "free_sulfur_dioxide": 3, 
        "total_sulfur_dioxide": 75, 
        "density": 1, 
        "pH": 3, 
        "sulphates": 1, 
        "alcohol": 9
    },

    "incorrect_col": {
        "fixed acidity": 5, 
        "volatile acidity": 1, 
        "citric acid": 0.5, 
        "residual sugar": 10, 
        "chlorides": 0.5, 
        "free sulfur dioxide": 3, 
        "total_sulfur dioxide": 75, 
        "density": 1, 
        "pH": 3, 
        "sulphates": 1, 
        "alcohol": 9
    }
}

TARGET_range = {
    "min": 3.0,
    "max": 8.0
}

def test_form_response_correct_range():
    res = form_response(input_data["correct_range"])
    assert TARGET_range["min"] <= float(res) <= TARGET_range["max"]

def test_api_response_correct_range():
    res = api_response(input_data["correct_range"])
    assert TARGET_range["min"] <= float(res["response"]) <= TARGET_range["max"]

def test_form_response_incorrect_range():
    with pytest.raises(prediction_service.prediction.NotInRange):
        _ = form_response(input_data["incorrect_range"])

def test_api_response_incorrect_range():
    res = api_response(input_data["incorrect_range"])
    assert res["response"] == prediction_service.prediction.NotInRange().message

def test_api_response_incorrect_col():
    res = api_response(input_data["incorrect_col"])
    assert res["response"] == prediction_service.prediction.NotInCols().message