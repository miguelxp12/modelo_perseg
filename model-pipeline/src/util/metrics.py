import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Any


class Metrics:

    @staticmethod
    def plot_bullseye(
        __df: pd.DataFrame
    ) -> Any:
        print(f"metrics plot_bullseye df: {__df.shape}")

        if 'PUP_PREDICT' in __df.columns:
            print("Column exists")
        else:
            print("doesnt exists")

        COL_DEMAND = "REAL_UNIDADES_VENDIDAS"

        df_predicted = __df.copy()
        df_predicted = df_predicted[(df_predicted['PUP_CALCULADO'] > 0)]

        y_name_to_predict = "PUP_PREDICT"
        df_predicted[y_name_to_predict] = np.where(df_predicted[y_name_to_predict].astype(float) < 0, 0.0,
                                                        df_predicted[y_name_to_predict].astype(float),)
        df_predicted["D_E"] = 0
        df_predicted.loc[(df_predicted[y_name_to_predict] <= 0.0) & (df_predicted.PUP_CALCULADO > 0.0), "D_E",] = 2.01
        df_predicted.loc[(df_predicted[y_name_to_predict] <= 0.0) & (df_predicted.PUP_CALCULADO == 0.0), "D_E",] = 9999
        df_predicted.loc[(df_predicted[y_name_to_predict] > 0.0), "D_E"] = (df_predicted.PUP_CALCULADO / df_predicted[y_name_to_predict])
        df_predicted["B_C"] = "No Aplica"
        df_predicted.loc[(df_predicted.D_E == 9999), "B_C"] = "No Aplica"
        df_predicted.loc[(df_predicted.D_E == 999), "B_C"] = "No estimado"
        df_predicted.loc[(df_predicted.D_E <= 0.3), "B_C"] = "0 - 0.3"
        df_predicted.loc[(df_predicted.D_E > 0.3) & (df_predicted.D_E <= 0.7), "B_C"] = "0.3 - 0.7"
        df_predicted.loc[(df_predicted.D_E > 0.7) & (df_predicted.D_E <= 1.3), "B_C"] = "0.7 - 1.3"
        df_predicted.loc[(df_predicted.D_E > 1.3) & (df_predicted.D_E <= 1.6), "B_C"] = "1.3 - 1.6"
        df_predicted.loc[(df_predicted.D_E > 1.6) & (df_predicted.D_E <= 2), "B_C"] = "1.6 - 2"
        df_predicted.loc[(df_predicted.D_E > 2), "B_C"] = ">2"
        df_wo_new_products = df_predicted.copy()
        # Graph
        a = df_wo_new_products.groupby(["B_C"], as_index=False).agg({COL_DEMAND: "sum"})
        a["porc_demanda"] = (a[COL_DEMAND] / sum(a[COL_DEMAND])) * 100
        a = df_wo_new_products.groupby(["B_C"], as_index=False).agg({COL_DEMAND: "sum"})
        a["porc_demanda"] = (a[COL_DEMAND] / sum(a[COL_DEMAND])) * 100

        plt.bar(a["B_C"], a["porc_demanda"], color="maroon", width=0.4)
        plt.ylim([0, 80])
        plt.grid(which="major", linewidth=1)
        plt.grid(which="minor", linewidth=0.2)

        # plt.show()

        # mape detallado start
        mape_detallado = sum(
            (
                abs(df_wo_new_products["PUP_CALCULADO"] - df_wo_new_products[y_name_to_predict])
                / df_wo_new_products["PUP_CALCULADO"]
            )
            * df_wo_new_products[COL_DEMAND]
        ) / (sum(df_wo_new_products[COL_DEMAND]))

        # mape detallado end
        print("mape cosmetics: ", mape_detallado)

        # bias detallado start
        bias_detallado = sum(
            (
                (df_wo_new_products["PUP_CALCULADO"] - df_wo_new_products[y_name_to_predict])
                / df_wo_new_products["PUP_CALCULADO"]
            )
            * df_wo_new_products[COL_DEMAND]
        ) / (sum(df_wo_new_products[COL_DEMAND]))
        # bias detallado end
        print("bias: ", bias_detallado)
        return plt

    @staticmethod
    def compute_aggregates(group):
        eps = 1e-19
        # Sum of columns
        sum_pup = group['PUP_CALCULADO'].sum()
        sum_pup_predict = group['PUP_PREDICT'].sum()
        total_units = group['REAL_UNIDADES_VENDIDAS'].sum()

        # BIN_MAYOR_2: Sum of REAL_UNIDADES_VENDIDAS for rows where pup/(pup_predict+eps) > 2,
        # divided by the total units, then add eps.
        bin_mayor_2_numer = group.loc[(group['PUP_CALCULADO'] / (group['PUP_PREDICT'] + eps)) > 2, 'REAL_UNIDADES_VENDIDAS'].sum()
        bin_mayor_2 = bin_mayor_2_numer / (total_units + eps) + eps

        # BIN_CENTRAL: Sum of REAL_UNIDADES_VENDIDAS for rows where the ratio is between 0.7 and 1.3,
        # divided by the total units, then add eps.
        bin_central_numer = group.loc[
            (group['PUP_CALCULADO'] / (group['PUP_PREDICT'] + eps) >= 0.7) &
            (group['PUP_CALCULADO'] / (group['PUP_PREDICT'] + eps) <= 1.3),
            'REAL_UNIDADES_VENDIDAS'
        ].sum()
        bin_central = bin_central_numer / (total_units + eps) + eps

        # MAPE: weighted mean absolute percentage error
        mape = (np.abs(group['PUP_CALCULADO'] - group['PUP_PREDICT']) / (group['PUP_CALCULADO'] + eps) * group['REAL_UNIDADES_VENDIDAS']).sum() / (total_units + eps)

        # Bias: weighted bias calculation
        bias = (((group['PUP_CALCULADO'] - group['PUP_PREDICT']) / (group['PUP_CALCULADO'] + eps)) * group['REAL_UNIDADES_VENDIDAS']).sum() / (total_units + eps)

        return pd.Series({
            'PUP_CALCULADO': sum_pup,
            'PUP_PREDICT': sum_pup_predict,
            'BIN_MAYOR_2': bin_mayor_2,
            'BIN_CENTRAL': bin_central,
            'MAPE': mape,
            'Bias': bias
        })
