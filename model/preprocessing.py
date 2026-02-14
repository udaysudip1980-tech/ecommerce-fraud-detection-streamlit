from sklearn.preprocessing import LabelEncoder, StandardScaler

# Drop columns that are not useful for fraud prediction
DROP_COLS = [
    "Transaction ID",
    "Customer ID",
    "IP Address",
    "Shipping Address",
    "Billing Address",
    "Transaction Date"
]

CATEGORICAL_COLS = [
    "Payment Method",
    "Product Category",
    "Customer Location",
    "Device Used"
]

def preprocess_dataframe(df, fit=False, encoders=None, scaler=None):
    df = df.drop(columns=DROP_COLS, errors="ignore") # Drop columns that are not useful for fraud prediction

    # Convert categorical features into numeric form
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
    else:
        for col, le in encoders.items():
            df[col] = df[col].apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            df[col] = le.transform(df[col])

    X = df.drop("Is Fraudulent", axis=1)
    y = df["Is Fraudulent"]

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X, X_scaled, y, encoders, scaler
