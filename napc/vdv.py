import numpy as np
import pandas as pd
from pathlib import Path
from .evaluation import grb_score
import math
import scipy.stats as stats

def eval_csv(res, alpha=0.05, delta=0.01, percent=True):

    """
    assumes df with all results
    computes equivalence test for counting accuracy (see https://www.vdv.de/457-v2.2-sds.pdfx?forced=true p. 73 and following)
    """

    #csv_path = Path(csv_path)
    #res = pd.read_csv(csv_path)#.query("n_frame<=1024")
    #res.dropna(inplace=True)

    # extract labels and predicitons, round predictions to integer
    p_b = np.round(res.b_pred)
    p_a = np.round(res.a_pred)
    n_b = res.n_boarding
    n_a = res.n_alighting

    # calculate metrics
    grb_b, grb_a = grb_score(p_b, n_b)/100, grb_score(p_a, n_a)/100

    n = len(res)

    # BS1 
    rel_diff_b= (p_b.sum() - n_b.sum()) / n_b.sum()
    rel_diff_a = (p_a.sum() - n_a.sum()) / n_a.sum()

    # BS2
    z = stats.norm.ppf(1-alpha/2)

    # BS3 3.1
    avg_n_b = n_b.mean()
    avg_n_a = n_a.mean()

    # BS3 3.2
    x_avg_a = (p_a-n_a).mean()
    x_avg_b = (p_b-n_b).mean()
    x_a = (p_a-n_a) 
    x_b = (p_b-n_b)
    stdbw_a = math.sqrt(((x_a - x_avg_a)**2).sum()/(n-1)) 
    stdbw_b = math.sqrt(((x_b - x_avg_b)**2).sum()/(n-1)) 

    # BS3 3.3
    stdbw_a = stdbw_a/avg_n_a
    stdbw_b = stdbw_b/avg_n_b 

    #print(z)
    #print(stdbw_a)

    # BS4
    intv_width_a = z * stdbw_a/(math.sqrt(n))
    intv_width_b = z * stdbw_b/(math.sqrt(n))

    # BS5
    intv_a = rel_diff_a-intv_width_a, rel_diff_a+intv_width_a
    intv_b = rel_diff_b-intv_width_b, rel_diff_b+intv_width_b

    # 
    if percent:
        intv_a = 100*intv_a[0], 100*intv_a[1]
        intv_b = 100*intv_b[0], 100*intv_b[1]
        delta = 100*delta

    #print(intv_a, intv_b)


    result = {
        "intv_a": intv_a, # confidence interval for relative difference in alighting
        "intv_b": intv_b, # confidence interval for relative difference in boarding
        "contained_a": abs(intv_a[0])<=delta and abs(intv_a[1])<=delta, # whether the confidence interval for alighting contains 0
        "contained_b": abs(intv_b[0])<=delta and abs(intv_b[1])<=delta, # whether the confidence interval for boarding contains 0
    }

    return result

def collect_csv_results(
    folder_path="/work/wassmer/napc_xlstm/output_folder/thesis/synced2"
):
    folder_path = Path(folder_path)
    results = []

    for csvfile in folder_path.rglob("*.csv"):
        try:
            df = pd.read_csv(csvfile)
            df["csv_name"] = csvfile.stem  # filename without .csv
            results.append(df)
        except Exception as e:
            print(f"Error processing {csvfile}: {e}")

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)
