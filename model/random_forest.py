from sklearn.ensemble import RandomForestClassifier

def build():
    return RandomForestClassifier(n_estimators=100, random_state=42)
