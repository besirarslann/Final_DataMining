import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df_train = pd.read_csv("train.csv")
df_train["Age"] = df_train["Age"] / 365
df_test = pd.read_csv("test.csv")
df_sub = pd.read_csv("sample_submission.csv")


def grab_col_names(dataframe, cat_th=10, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    return cat_cols, num_cols, cat_but_car


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


def target_summary_with_cat(dataframe: object, target: object, categorical_col: object) -> object:
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


def correlation_matrix(df, cols):
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(df[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name, q1=0.05, q3=0.95):
    if dataframe[col_name].dtype.name == 'category':
        return False  # Skip categorical columns for outlier check
    else:
        low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
        outliers = (dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)
        return outliers.any()


def plot_categorical_relationships(data, cat_target_col, num_cols):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    # Violinplot
    sns.violinplot(x=cat_target_col, y=num_cols, data=data, inner="quartile")
    plt.title(f"{cat_target_col} vs {num_cols}")
    plt.show()


cat_cols, num_cols, cat_but_car = grab_col_names(df_train, cat_th=5, car_th=20)

num_cols = [col for col in num_cols if col != "id"]

for col in cat_cols:
    cat_summary(df_train, col, plot=True)

for col in num_cols:
    num_summary(df_train, col, plot=True)

for col in num_cols:
    target_summary_with_num(df_train, "Status", col)

correlation_matrix(df_train, num_cols)

for col in num_cols:
    sns.boxplot(data=df_train[col])
    plt.title(f'Boxplot for {col}')
    plt.show()

for col in num_cols:
    print(outlier_thresholds(df_train, col))

for col in num_cols:
    print(check_outlier(df_train, col))

for col in num_cols:
    plot_categorical_relationships(df_train, "Status", col)
