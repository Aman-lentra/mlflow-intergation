import mlflow.sklearn
import numpy as np

# Load the model from MLflow
model_uri = "runs:/a48fbf004b4b47819ee2af3830e02aa0/model"
loaded_model = mlflow.sklearn.load_model(model_uri)

# Create a new test dataset
new_data = np.array([[0.03807591, 0.05068012, 0.06169621, 0.02187235, -0.0442235, -0.03482076, -0.04340085, -0.00259226, 0.01990842, -0.01764613],
                     [-0.00188202, -0.04464164, -0.05147406, -0.02632783, -0.00844872, 0.00720652, 0.06336665, -0.03949338, -0.06832974, -0.09220405]])

# Make predictions on the new test data
predictions = loaded_model.predict(new_data)

# Print the predictions
print(predictions)
