# Action_reco_using_conformal_pred
Action recognition along with conformal prediction using assembly101 data and USING TSM pretrained

**Below are the high-level steps you’d follow to implement the prediction analysis for your TSM model:**

Prepare Your Environment:

Ensure you have Python 3, NumPy, and PyTorch installed.
Place your generated prediction file (pred.npy) (and optionally the ground truth file, e.g. gt.npy) in your working directory.
Generate Predictions (if not already done):

Run the provided TSM evaluation script (e.g. test_models.py) with your validation/test data and pretrained weights to produce the pred.npy file.
Create a Prediction Analysis Script:

Write a new Python script (e.g. inspect_predictions.py).
In the script, load pred.npy using numpy.load().
Print out the shape of the predictions array and a sample of the predicted values.
(Optional) If you have ground truth labels, load them and compute the accuracy or confusion matrix.
Run and Review:

Execute your prediction analysis script from the command line.
Review the printed output (e.g. the prediction distribution and any computed metrics) to understand how well the model is performing.

