import mlflow
import mlflow.sklearn
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the diabetes dataset
diabetes = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(diabetes.data, diabetes.target, random_state=42)

# Train your linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Log your model with MLflow
with mlflow.start_run():
    # Log the model parameters
    mlflow.log_param("normalize", model.normalize)

    # Log the model itself
    mlflow.sklearn.log_model(model, "model")

    # Log the evaluation metrics
    train_rmse = model.score(X_train, y_train)
    test_rmse = model.score(X_test, y_test)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("test_rmse", test_rmse)
