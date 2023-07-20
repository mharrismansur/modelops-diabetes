
from sklearn import metrics
from teradataml import DataFrame, copy_to_sql
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import joblib
import json
import numpy as np
import pandas as pd


def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()

    model = joblib.load(f"{context.artifact_input_path}/model.joblib")

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]

    test_df = DataFrame.from_query(context.dataset_info.sql)
    test_pdf = test_df.to_pandas(all_rows=True).reset_index() # added reset_index()

    X_test = test_pdf[feature_names]
    y_test = test_pdf[target_name]

    print("Scoring")
#     print(test_pdf)
    y_pred = model.predict(X_test)

    y_pred_tdf = pd.DataFrame(y_pred, columns=[target_name])
    y_pred_tdf["PI"] = test_pdf["PI"].values

    evaluation = {
        'R2-Score': '{:.2f}'.format(metrics.r2_score(y_test, y_pred))
    }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)

#     metrics.plot_confusion_matrix(model, X_test, y_test)
#     save_plot('Confusion Matrix', context=context)

#     metrics.plot_roc_curve(model, X_test, y_test)
#     save_plot('ROC Curve', context=context)

#     # xgboost has its own feature importance plot support but lets use shap as explainability example
#     import shap

#     shap_explainer = shap.TreeExplainer(model['xgb'])
#     shap_values = shap_explainer.shap_values(X_test)

#     shap.summary_plot(shap_values, X_test, feature_names=feature_names,
#                       show=False, plot_size=(12, 8), plot_type='bar')
#     save_plot('SHAP Feature Importance', context=context)

#     feature_importance = pd.DataFrame(list(zip(feature_names, np.abs(shap_values).mean(0))),
#                                       columns=['col_name', 'feature_importance_vals'])
#     feature_importance = feature_importance.set_index("col_name").T.to_dict(orient='records')[0]

    predictions_table = "diabetes_predictions_tmp"
    copy_to_sql(df=y_pred_tdf, table_name=predictions_table, index=False, if_exists="replace", temporary=True)

    record_evaluation_stats(features_df=test_df,
                            predicted_df=DataFrame.from_query(f"SELECT * FROM {predictions_table}"),
#                             importance=feature_importance,
                            context=context)
