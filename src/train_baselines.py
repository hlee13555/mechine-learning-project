import mlflow
import mlflow.sklearn
import pandas as pd
from features import preprocess
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/val.csv")
test = pd.read_csv("data/processed/test.csv")

# Preprocessing
Xtrain, scaler = preprocess(train)
Xval, _ = preprocess(val, scaler)
Xtest, _ = preprocess(test, scaler)

ytrain = train["is_efficient"]
yval = val["is_efficient"]
ytest = test["is_efficient"]

ytrainR = train["Y1"]
yvalR = val["Y1"]
ytestR = test["Y1"]

mlflow.set_experiment("energy_efficiency_baselines")

# --- Classification Baselines ---
for modelName, model in {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "DecisionTreeClassifier": DecisionTreeClassifier(max_depth=5)
}.items():
    with mlflow.start_run(run_name=f"class_{modelName}"):
        model.fit(Xtrain, ytrain)
        preds = model.predict(Xtest)
        probs = model.predict_proba(Xtest)[:, 1]

        acc = accuracy_score(ytest, preds)
        f1 = f1_score(ytest, preds)
        auc = roc_auc_score(ytest, probs)

        mlflow.log_metrics({"accuracy": acc, "f1": f1, "roc_auc": auc})
        mlflow.sklearn.log_model(model, f"{modelName}")
        print(f"{modelName} -> acc={acc:.3f}, f1={f1:.3f}, auc={auc:.3f}")

# --- Regression Baselines ---
for modelName, model in {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(max_depth=5)
}.items():
    with mlflow.start_run(run_name=f"reg_{modelName}"):
        model.fit(Xtrain, ytrainR)
        preds = model.predict(Xtest)

        mae = mean_absolute_error(ytestR, preds)
        rmse = mean_squared_error(ytestR, preds) ** 0.5

        mlflow.log_metrics({"mae": mae, "rmse": rmse})
        mlflow.sklearn.log_model(model, f"{modelName}")
        print(f"{modelName} -> MAE={mae:.3f}, RMSE={rmse:.3f}")
