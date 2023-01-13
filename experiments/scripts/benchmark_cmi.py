import collections

import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from dodiscover.ci import CMITest
from dodiscover.ci.simulate import nonlinear_additive_gaussian

import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_cmi(data):
    result_dict = collections.defaultdict(list)
    cmi_est = CMITest(k=5, n_shuffle_nbrs=10)

    if "Z" in data.columns:
        z_covariates = {"Z"}
    else:
        z_covariates = {col for col in data.columns if col.startswith("Z_")}

    # compute the estimate of the CMI
    val = cmi_est._compute_cmi(data, "X", "Y", set())
    result_dict["cmi_null"].append(val)

    # compute the estimate of the CMI
    val = cmi_est._compute_cmi(data, "X", "Y", z_covariates)
    result_dict["cmi_z"].append(val)

    # compute the estimate of the CMI
    val = cmi_est._compute_cmi(data, "X", "Y", {"A"})
    result_dict["cmi_a"].append(val)

    # compute the estimate of the CMI
    val = cmi_est._compute_cmi(data, "X", "Y", {"B"})
    result_dict["cmi_b"].append(val)

    # compute the estimate of the CMI
    val = cmi_est._compute_cmi(data, "X", "Y", {"C"})
    result_dict["cmi_c"].append(val)

    # compute the estimate of the CMI
    val = cmi_est._compute_cmi(data, "X", "Y", {"noise_var"})
    result_dict["cmi_noise_var"].append(val)

    # create resulting dataframe
    result_df = pd.DataFrame.from_dict(result_dict)
    return result_df


if __name__ == '__main__':
    rng = np.random.default_rng()
    seed = 12345
    n_repeats = 5

    def linear_func(x):
        return x
    
    def cube_func(x):
        return np.power(x, 3)

    nonlinear_funcs = [linear_func, np.cos, np.sin, np.square, cube_func, np.tanh]
    dims_zs = np.arange(1, 12, 2)
    stds = np.arange(0.1, 1.5, 0.2)
    sample_sizes = np.linspace(500, 5000, 10).astype(int)

    overall_res_df = pd.DataFrame()
    jdx = -1

    for nonlinear_func in tqdm(nonlinear_funcs, desc='outer', position=0):
        for dims_z in dims_zs:
            for std in stds:
                for n_samples in sample_sizes:
                    # overall_res_df = pd.DataFrame()
                    jdx += 1
                    # if jdx < 14:
                    #     continue

                    for idx in range(n_repeats):
                        seed += idx
                        # generate A as independent
                        A, _, _ = nonlinear_additive_gaussian(
                            "ind",
                            n_samples=n_samples,
                            random_state=seed,
                            nonlinear_func=nonlinear_func,
                            dims_z=dims_z,
                            std=std,
                        )

                        # A -> X -> Y <- Z
                        X, Y, Z = nonlinear_additive_gaussian(
                            "dep",
                            n_samples=n_samples,
                            cause_var_x=A,
                            random_state=seed,
                            nonlinear_func=nonlinear_func,
                            dims_z=dims_z,
                            std=std,
                        )
                        # X -> B; Y -> C
                        B, C, _ = nonlinear_additive_gaussian(
                            "ind",
                            n_samples=n_samples,
                            cause_var_x=X,
                            cause_var_y=Y,
                            random_state=seed,
                            nonlinear_func=nonlinear_func,
                            dims_z=dims_z,
                            std=std,
                        )

                        df = pd.DataFrame()
                        for col_name, arr in zip(
                            ["A", "B", "C", "X", "Y", "Z"], [A, B, C, X, Y, Z]
                        ):
                            columns = [f"{col_name}_{idx}" for idx in range(arr.shape[1])]
                            if len(columns) == 1:
                                columns = [col_name]
                            _df = pd.DataFrame(arr, columns=columns)
                            df = pd.concat((df, _df), axis=1)
                        df["noise_var"] = rng.normal(size=(n_samples,))

                        result_df = evaluate_cmi(df)
                        result_df["nonlinear_func"] = nonlinear_func.__name__
                        result_df["dims_z"] = dims_z
                        result_df["std"] = std
                        result_df["idx"] = idx

                        overall_res_df = pd.concat((overall_res_df, result_df), axis=0)

                # fname = f"~/Downloads/cmi_results/cmi_{jdx}.csv"
                # overall_res_df.to_csv(fname)
        #         break
        #     break
        # break
    fname = f"~/Downloads/cmi_results/cmi_funcs_stds_dimsz.csv"
    overall_res_df.to_csv(fname)