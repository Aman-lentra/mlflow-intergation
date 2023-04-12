import mlflow.sklearn

# Load the model from MLflow
model_uri = "runs:/270303ad72644354af53fba1b12eabcc/model"
model = mlflow.sklearn.load_model(model_uri)

# Export the model as a directory
mlflow.sklearn.save_model(model, "model_export")
