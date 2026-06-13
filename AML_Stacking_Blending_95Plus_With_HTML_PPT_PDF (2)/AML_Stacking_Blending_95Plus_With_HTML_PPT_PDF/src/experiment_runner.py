from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

from utils import configure_environment, load_wine_quality, build_preprocessor, stratified_split, timed_fit
from evaluation import evaluate_classifier, results_table
from stacking_from_scratch import CustomStackingClassifier
from blending import CustomBlendingClassifier

SEED = 42

def wrap(model, X):
    return Pipeline([("prep", build_preprocessor(X)), ("model", model)])

def build_base_learners(X, seed=42):
    learners = [
        ("lr", wrap(LogisticRegression(max_iter=5000, random_state=seed), X)),
        ("svm", wrap(SVC(kernel="rbf", probability=True, random_state=seed), X)),
        ("knn", wrap(KNeighborsClassifier(n_neighbors=9), X)),
        ("rf", wrap(RandomForestClassifier(n_estimators=250, random_state=seed, n_jobs=-1), X)),
        ("gb", wrap(GradientBoostingClassifier(random_state=seed), X)),
    ]
    if XGBClassifier is not None:
        learners.append(("xgb", wrap(XGBClassifier(n_estimators=250, learning_rate=0.05, max_depth=4, eval_metric="logloss", random_state=seed), X)))
    if LGBMClassifier is not None:
        learners.append(("lgbm", wrap(LGBMClassifier(n_estimators=250, learning_rate=0.05, random_state=seed, verbose=-1), X)))
    return learners

if __name__ == "__main__":
    configure_environment(SEED)
    X, y, meta = load_wine_quality("red")
    X_train, X_test, y_train, y_test = stratified_split(X, y, seed=SEED)

    base = build_base_learners(X, SEED)
    models = {name: model for name, model in base}
    models["hard_voting"] = VotingClassifier(estimators=base, voting="hard", n_jobs=-1)
    models["soft_voting"] = VotingClassifier(estimators=base, voting="soft", n_jobs=-1)
    models["custom_stacking"] = CustomStackingClassifier(base, n_splits=5, random_state=SEED)
    models["sklearn_stacking"] = StackingClassifier(base, final_estimator=LogisticRegression(max_iter=5000), cv=5, stack_method="predict_proba", n_jobs=-1)
    models["custom_blending"] = CustomBlendingClassifier(base, blend_size=0.25, random_state=SEED)

    rows = []
    for name, model in models.items():
        print(f"Training {name}")
        t = timed_fit(model, X_train, y_train)
        rows.append(evaluate_classifier(model, X_test, y_test, name, t))
    print(results_table(rows))
