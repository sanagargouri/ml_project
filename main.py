from model_pipeline import (
    load_data,
    prepare_data,
    split_data,
    train_model,
    predict,
    evaluate_model,
    save_model,
    load_model
)


# 1. Chargement des données
df = load_data("Churn_Modelling.csv")

# 2. Préparation des données
X, y = prepare_data(df)

# 3. Séparation des données
X_train, X_test, y_train, y_test = split_data(X, y)

# 4. Entraînement du modèle
model = train_model(X_train, y_train)

# 5. Prédictions
y_pred = predict(model, X_test)

# 6. Évaluation
accuracy, cm = evaluate_model(y_test, y_pred)

print("Accuracy :", accuracy)
print("Matrice de confusion :")
print(cm)

# 7. Sauvegarde du modèle
save_model(model, "churn_model.pkl")

print("Modèle sauvegardé avec succès.")

# 8. Test du chargement du modèle
loaded_model = load_model("churn_model.pkl")

print("Modèle chargé :")
print(loaded_model)