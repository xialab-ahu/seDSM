import operator
import pickle
import numpy as np


class Pipeline(object):
    def __init__(self, raw_pipelines):
        self.pipelines = raw_pipelines

    def transform(self, data, *args):
        out = np.zeros_like(data)
        for p in self.pipelines:
            out += p.transform(data)

        out = out / len(self.pipelines)
        return out


class CVEnsemble(object):
    def __init__(self, data_pipeline, models_pool, selection_index):
        self.models_pool = models_pool
        self.selection_index = selection_index
        self.data_pipeline = data_pipeline

    def predict_proba(self, data, n_estimator=42):
        # Fill na values
        data = self.data_pipeline.transform(data)

        predictions = []
        for i in range(len(self.models_pool)):
            index = self.selection_index[i]
            # Select top n_estimator models to integrate.
            op = operator.itemgetter(*index[: n_estimator])
            selected_models_cv = op(self.models_pool[f"cv-{i}"])
            predictions_ = np.array([m.predict_proba(data)[:, 1] for m in selected_models_cv]).mean(0)
            predictions.append(predictions_)

        # Integrate selective ensemble model from experiments of five times.
        predictions = np.array(predictions).mean(0)
        predictions = np.vstack([1 - predictions, predictions]).T
        return predictions

    @staticmethod
    def from_dir(model_dir):
        models_pool = pickle.load(open(r"./model/model_pools.pickle", "rb"))
        selection_index = pickle.load(open(
            r"./model/ratio_errors_sorted_index_0.pickle", "rb"))
        raw_pipeline = pickle.load(open(
            r"./model/pipelines.pkl", "rb"))
        data_pipeline = Pipeline(raw_pipeline)
        cv_ensemble = CVEnsemble(data_pipeline, models_pool, selection_index)
        return cv_ensemble











