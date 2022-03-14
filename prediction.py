from model import CVEnsemble
import pandas as pd
import argparse
import numpy as np
import os


argparser = argparse.ArgumentParser()
argparser.add_argument("--csv", type=str, default="./example/example.csv")
argparser.add_argument("--output_dir", type=str, default="./results")
args = argparser.parse_args()

na_values = ["na", "nan", "Na", "NaN"]

data = pd.read_csv(args.csv, sep="\t", na_values=na_values)
model = CVEnsemble.from_dir("./model")


# Get predictions of seDSM.
prediction = model.predict_proba(data)

# Save predictions to `./results`
if not os.path.exists(args.otuput_dir):
    os.makdirs(args.output_dir)
np.savetxt(os.path.join(args.output_dir, "results.csv"), prediction, fmt="%4.3f")
