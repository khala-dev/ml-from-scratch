# Linear Regression from Scratch

<!-- 1-2 sentences: what is in this folder, what did you learn? -->

## 1. The idea

<!-- In your words: what do we want to predict, and from what? -->

## 2. Hypothesis (the line)

$$\hat{y} = \theta_0 + \theta_1 x$$

<!-- What does each symbol mean? What does theta0 do? What does theta1 do? -->

## 3. Cost function

$$J(\theta_0, \theta_1) = \frac{1}{2N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2$$

<!-- Why do we square the errors? Why do we divide by N? -->

## 4. Gradient descent

<!-- The 5 steps in order: predict, errors, gradients, update, repeat. Say where each result goes. -->

## 5. Learning rate

<!-- What happens when it is too small? Too large? Use the numbers from the experiment. -->

## 6. Stopping criteria

<!-- Why do we need tolerance? Do we reach the exact minimum? -->

## 7. Worked example (apartments)

<!-- Put the data table and the first iterations (cost: 28125, 4512.5, 2170.9...) -->

## 8. How to run

```bash
pip install numpy pytest matplotlib
cd linear_regression
pytest
```

## 9. What I still don't understand

<!-- Honest list. It helps you and shows your real level. -->