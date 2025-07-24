# tools/analysis_tools.py

import matplotlib
matplotlib.use("Agg")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.tools import tool
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
from io import BytesIO
import base64
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import base64
from typing import List, Any
import joblib
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from category_encoders.target_encoder import TargetEncoder
from agents.modelers.random_forest_agent import RandomForestAgent
from agents.modelers.logistic_regression_agent import LogisticRegressionAgent
from agents.modelers.svm_agent import SVMClassifierAgent
from agents.modelers.decision_tree_agent import DecisionTreeAgent
from tools.common_utils import fig_to_base64, create_error_plot
import os


# Converts any matplotlib figure to a base64-encoded PNG string
def fig_to_base64(fig) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string with basic error handling."""
    try:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        return img_base64
    except Exception as e:
        return f"Error encoding figure: {str(e)}"
    finally:
        plt.close(fig)


def safe_tight_layout(fig, 
                      left=0.15, right=0.95, top=0.95, bottom=0.25,
                      pad=1.0, w_pad=None, h_pad=None):
    """
    Try fig.tight_layout(), but if it blows up, fall back to fixed margins.
    """
    try:
        fig.tight_layout(pad=pad, w_pad=w_pad, h_pad=h_pad)
    except Exception:
        fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)


@tool
def basic_summary(data: pd.DataFrame) -> str:
    """
    Returns a basic summary of the dataset.

    Args:
        data: Pandas DataFrame

    Returns:
        String containing shape, column types, and head of the dataset.
    """
    print("********************************************")
    print("Generating basic summary of the dataset...")
    print("********************************************")
    return f"Shape: {data.shape}\n\nColumns:\n{data.dtypes}\n\nHead:\n{data.head()}"

#------------------------------------- Zainab >> Updated the function to return a dictionary instead of a string
@tool
def missing_values_report(data: pd.DataFrame) -> dict:
    """
    Reports missing values per column.

    Args:
        data: Pandas DataFrame

    Returns:
        String with column-wise null counts.
    """
    print("********************************************")
    print("missing_values_report called")
    print("********************************************")
    # Initialize an empty dictionary to store the NaN counts for each column
    nan_counts = {}

    # Loop through each column and count NaN values
    for column in data.columns:
        nan_counts[column] = data[column].isnull().sum()

    return nan_counts


@tool
def frequency_distribution(data: pd.DataFrame, column: str) -> dict:
    """
    Shows frequency of each category in a column.

    Args:
        data: DataFrame
        column: Target column (categorical or otherwise)

    Returns:
        Frequency table as string.
    """
    print("********************************************")
    print(f"frequency_distribution called for column: {column}")
    print("********************************************")
     # Initialize an empty dictionary to store the NaN counts for each column
    value_counts = {}

    # Loop through each column and count NaN values
    for column in data.columns:
        value_counts[column] = data[column].value_counts(dropna=False)

    return value_counts

#_______________________ Zainab >> Patch the Tool to Return Proper Format: Apply the Fix to All Visualization Tools
# Wherever we're using fig_to_base64(fig), wrap it with:f"data:image/png;base64,{...}"
# @tool
# def barplot_counts(data: pd.DataFrame, column: str) -> str:
#     """
#     Generates a horizontal barplot showing count per category.

#     Args:
#         data: DataFrame
#         column: Target column (categorical)

#     Returns:
#         Base64-encoded PNG string of the plot.
#     """
#     fig, ax = plt.subplots(figsize=(10, 6))
#     sns.countplot(data=data, y=column, order=data[column].value_counts().index, ax=ax)
#     ax.set_title(f"Incident Counts by {column}")
    
#     print("********************************************")
#     print(f"barplot_counts called for column: {column}")
#     print("********************************************")
    # return f"data:image/png;base64,{fig_to_base64(fig)}"
 #--------------------- Zainab >> Updated the barplot_counts function to show top N categories
@tool
def barplot_counts(
    data: pd.DataFrame,
    column: str,
    top_n: int = 10
) -> str:
    """
    Generates a horizontal barplot showing the top N counts for `column`.

    Args:
        data: Pandas DataFrame
        column: Target column (categorical)
        top_n: Number of top categories to display

    Returns:
        Base64-encoded PNG of the plot.
    """
    # 1) compute top-N
    counts = data[column].value_counts(dropna=False)
    top    = counts.nlargest(top_n)
    categories = top.index.tolist()
    values     = top.values

    # 2) get a Viridis color for each bar
    colors = sns.color_palette("viridis", len(categories))

    # 3) plot with matplotlib directly
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(categories, values, color=colors)

    # 4) styling
    ax.set_title(f"Top {top_n} {column} Counts", fontsize=14, pad=10)
    ax.set_xlabel("Count", fontsize=12)
    ax.set_ylabel(column, fontsize=12)
    ax.grid(True, axis='x', alpha=0.3)

    # 5) annotate bars
    for i, val in enumerate(values):
        ax.text(
            val + val * 0.01,    # little offset to the right
            i,                   # y-position
            str(val),
            va="center",
            fontsize=10
        )

    # 6) layout & export
    safe_tight_layout(fig, left=0.3)
    png = fig_to_base64(fig)
    plt.close(fig)
    return f"data:image/png;base64,{png}"



@tool
def distribution_plot(
    data: pd.DataFrame,
    column: str,
    trim_percentile: float = 0.99,
    dpi: int = 150
) -> str:
    """
    Plots a trimmed, log-scaled histogram with KDE for a numeric column.

    - Clips values above the `trim_percentile` quantile to that threshold.
    - Applies log-scaling on the x-axis to spread out heavy tails.
    Args:
         data: DataFrame
         column: Target numeric column
         trim_percentile: Quantile to trim at (default 0.99)
         dpi: DPI for the plot (default 150)

     Returns:
         Base64-encoded image of the plot.
    """
    # 1) Drop nulls and compute clip threshold
    vals = data[column].dropna()
    if vals.empty:
        return create_error_plot(f"No data in column '{column}' to plot.")

    cap = vals.quantile(trim_percentile)
    vals_clipped = vals.clip(upper=cap)

    # 2) Build figure
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    
    # 3) Plot clipped data with log-scale
    sns.histplot(
        vals_clipped,
        kde=True,
        ax=ax,
        bins=50
    )
    ax.set_xscale('log')
    
    # 4) Titles & labels
    ax.set_title(f"Trimmed & Log-Scaled Distribution of {column}", fontsize=16, pad=12)
    ax.set_xlabel(column, fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.tick_params(labelsize=12)
    
    # 5) Indicate what we clipped
    ax.text(
        0.95, 0.95,
        f"Values > {cap:.2f} clipped",
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=10, color='gray'
    )
    
    # rotate labels and apply safe tight layout
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    safe_tight_layout(fig, bottom=0.3, left=0.2)
    return f"data:image/png;base64,{fig_to_base64(fig)}"


# Duplicate definition removed. Use the fig_to_base64 function defined earlier in the file.

@tool
def boxplot(data: pd.DataFrame, column: str) -> str:
    """
    Boxplot with outlier filtering and mean indicator.

    Args:
        data: DataFrame
        column: numeric column name

    Returns:
        Base64 string of the plot
    """
    series = data[column].dropna()

    # IQR filter to remove extreme outliers
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    filtered = series[(series >= lower) & (series <= upper)]

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(x=filtered, ax=ax, color='skyblue', fliersize=2)  # Smaller outliers
    mean_val = filtered.mean()
    ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
    ax.set_title(f"Boxplot of {column} (IQR Filtered)")
    ax.set_xlabel(column)
    ax.legend()

    return f"data:image/png;base64,{fig_to_base64(fig)}"


#------------------------- Zainab ------------------------------------
#Clip extreme values at the 99th percentile (so a few huge outliers don’t stretch the plot).
#Optionally apply a log-scale on the value axis to spread out the distribution.
#Increase figure size and dpi for crispness.
#Rotate the boxplot sideways for easier reading of the y-axis label.
#Annotate the plot with the clip threshold.

@tool
def define_outliers(data: pd.DataFrame, column: str) -> str:
    """
    Detects outliers in a numeric column using the IQR method.

    Args:
        data: DataFrame
        column: Target numeric column

    Returns:
        A message summarizing the number of outliers detected.
    """
    if not pd.api.types.is_numeric_dtype(data[column]):
        return f"Column '{column}' is not numeric"
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    count = ((data[column] < lower) | (data[column] > upper)).sum()
    
    print("********************************************")
    print(f"define_outliers called for column: {column}")
    print("********************************************")
    return f"{count} outliers detected in '{column}' using IQR"


# @tool
# def correlation_heatmap(data: pd.DataFrame) -> str:
#     """
#     Generates a correlation heatmap for numeric features.

#     Args:
#         data: DataFrame

#     Returns:
#         Base64-encoded PNG heatmap of correlations.
#     """
#     fig, ax = plt.subplots(figsize=(10, 8))
#     sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True, cmap='coolwarm', ax=ax)
#     ax.set_title("Correlation Matrix")
    
#     print("********************************************")
#     print("correlation_heatmap called")
#     print("********************************************")
#     return f"data:image/png;base64,{fig_to_base64(fig)}"

@tool
def correlation_heatmap(
    data: pd.DataFrame,
    max_vars: int = 15
) -> str:
    """
    Full correlation heatmap of the top‐variance numeric columns.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1) Pick numeric & top‐variance cols
    num_df = data.select_dtypes(include=np.number)
    if num_df.shape[1] < 2:
        return create_error_plot("Not enough numeric columns for correlation heatmap")

    variances = num_df.var().sort_values(ascending=False)
    cols = variances.head(max_vars).index.tolist()
    corr = num_df[cols].corr()
 # 2) Plot full matrix (no mask)
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 8},
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        ax=ax
    )
    # 4) Styling
    ax.set_title(f"Correlation Matrix (top {len(cols)} vars)", fontsize=18, pad=12)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=10)

    # 5) Layout
    safe_tight_layout(fig, bottom=0.3, left=0.2)

    # 6) Export
    png = fig_to_base64(fig)
    plt.close(fig)
    return f"data:image/png;base64,{png}"


from typing import List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from langchain.tools import tool
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def fig_to_base64(fig) -> str:
    """Helper to convert a matplotlib figure to base64 PNG."""
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return img_b64

@tool
def cluster_kmeans(
    data: pd.DataFrame,
    features: list,
    min_k: int = 2,
    max_k: int = 10,
    max_plot_points: int = 5000
) -> str:
    """
    KMeans clustering with:
      - auto-k selection by silhouette score
      - log1p transform of numeric features
      - PCA → 2D for visualization
      - sampling up to max_plot_points for plotting
      - distinct colors + black X centroids

    Returns a base64 PNG data‐URI.
    """
    try:
        # --- 1) Prepare the feature matrix ---
        df_work = data[features].copy()
        proc_feats = []
        for f in features:
            if pd.api.types.is_datetime64_any_dtype(df_work[f]):
                df_work[f + "_days"] = (df_work[f] - df_work[f].min()).dt.days
                proc_feats.append(f + "_days")
            elif pd.api.types.is_numeric_dtype(df_work[f]):
                proc_feats.append(f)
            else:
                df_work[f + "_enc"] = pd.Categorical(df_work[f]).codes
                proc_feats.append(f + "_enc")

        # log1p for all numeric to reduce skew/outliers
        df_work[proc_feats] = df_work[proc_feats].apply(np.log1p)

        X = df_work[proc_feats].fillna(0).values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # --- 2) Auto-select k via silhouette ---
        from sklearn.metrics import silhouette_score
        best_k, best_score, best_model = min_k, -1.0, None
        for k in range(min_k, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xs)
            score = silhouette_score(Xs, km.labels_)
            if score > best_score:
                best_k, best_score, best_model = k, score, km

        labels = best_model.labels_
        centers = best_model.cluster_centers_

        # --- 3) PCA → 2D for plotting ---
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        Xpca = pca.fit_transform(Xs)
        CenPCA = pca.transform(centers)
        var1, var2 = pca.explained_variance_ratio_

        # --- 4) Sample for plotting if too many points ---
        n = Xpca.shape[0]
        if n > max_plot_points:
            rng = np.random.RandomState(42)
            idx = rng.choice(n, max_plot_points, replace=False)
        else:
            idx = np.arange(n)
        X_plot = Xpca[idx]
        lbl_plot = labels[idx]

        # --- 5) Plot everything ---
        fig, ax = plt.subplots(figsize=(10, 8))
        palette = sns.color_palette("tab10", best_k)

        for cluster in range(best_k):
            mask = lbl_plot == cluster
            ax.scatter(
                X_plot[mask, 0], X_plot[mask, 1],
                s=20, alpha=0.6,
                color=palette[cluster],
                label=f"Cluster {cluster+1}"
            )

        # centroids
        ax.scatter(
            CenPCA[:, 0], CenPCA[:, 1],
            marker="X", s=200,
            c="black", linewidths=2,
            label="Centroids"
        )

        ax.set_xlabel(f"PC1 ({var1*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({var2*100:.1f}% var)")
        ax.set_title(f"KMeans Clustering (k={best_k}, silhouette={best_score:.2f})")
        ax.legend(loc="best")
        plt.tight_layout()

        return f"data:image/png;base64,{fig_to_base64(fig)}"

    except Exception as e:
        return create_error_plot(f"cluster_kmeans failed: {e}")
from matplotlib import pyplot as plt
import base64
from io import BytesIO


@tool
def pca_projection(df, cat_features=None, num_features=None, sample_size=5000, n_components=2, top_features=10) -> str:
    """
    Generate a PCA biplot from a DataFrame and return base64-encoded image.
    """
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA

    # Auto-detect features if not provided
    if cat_features is None:
        cat_features = df.select_dtypes(include='object').columns.tolist()
    if num_features is None:
        num_features = df.select_dtypes(include='number').columns.tolist()

    selected_columns = cat_features + num_features
    df_clean = df[selected_columns].dropna()

    # Sampling
    if len(df_clean) > sample_size:
        df_clean = df_clean.sample(n=sample_size, random_state=42)

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_features),
            ('num', StandardScaler(), num_features)
        ]
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=n_components))
    ])

    # Fit-transform and extract PCA info
    X_pca = pipeline.fit_transform(df_clean)
    pca_model = pipeline.named_steps['pca']
    vectors = pca_model.components_.T
    var_ratio = pca_model.explained_variance_ratio_
    feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()

    # Plot using fig/ax to capture for base64
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.3, color='steelblue')
    ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}% var)')
    ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}% var)')
    ax.set_title('PCA Biplot')

    magnitudes = np.sum(vectors**2, axis=1)
    top_idx = np.argsort(magnitudes)[-top_features:]

    for i in top_idx:
        ax.arrow(0, 0, vectors[i, 0], vectors[i, 1], color='orangered', alpha=0.7, head_width=0.05)
        ax.text(vectors[i, 0]*1.2, vectors[i, 1]*1.2, feature_names[i], ha='center', fontsize=9, color='darkred')

    ax.grid(True)
    fig.tight_layout()

    return f"data:image/png;base64,{fig_to_base64(fig)}"


#------------ zainab >> Updated the frequency parameter from 'M' to 'ME' for month-end
@tool
def time_series_decomposition(
    data: pd.DataFrame,
    date_col: str,
    value_col: str,
    freq: str = 'ME',
    period: int = 12
) -> str:
    """
    Decomposes a time series into observed, trend, seasonal, and residual components.

    Args:
        data: DataFrame containing the series.
        date_col: Name of the date column.
        value_col: Numeric column to decompose.
        freq: Resampling frequency (e.g., 'D','M','ME').
        period: Seasonal period for decomposition (e.g., 12 for monthly data).

    Returns:
        Base64‐encoded PNG of the 2×2 decomposition plot.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from statsmodels.tsa.seasonal import seasonal_decompose

    # 1) Prepare time series
    ts = (
        data
        .assign(_date=lambda df: pd.to_datetime(df[date_col]))
        .set_index('_date')[value_col]
        .resample(freq)
        .sum()
        .dropna()
    )
    if ts.empty or len(ts) < period * 2:
        return create_error_plot("Time series too short for decomposition")

    # 2) Perform decomposition
    result = seasonal_decompose(ts, model='additive', period=period)

    # 3) Build custom 2×2 figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100, sharex=True)
    components = {
        "Observed":    result.observed,
        "Trend":       result.trend,
        "Seasonal":    result.seasonal,
        "Residual":    result.resid
    }

    for ax, (title, series) in zip(axes.flatten(), components.items()):
        ax.plot(series.index, series.values, linewidth=1)
        ax.set_title(title, fontsize=14, pad=6)
        ax.grid(True, alpha=0.3)
        if title in ("Trend", "Observed"):
            ax.set_ylabel(value_col, fontsize=12)
        else:
            ax.set_ylabel(title, fontsize=12)

    # 4) X‐axis formatting
    plt.setp(axes[1, 0].get_xticklabels(), rotation=45, ha="right")
    plt.setp(axes[1, 1].get_xticklabels(), rotation=45, ha="right")

    # 5) Layout
    safe_tight_layout(fig, bottom=0.15, h_pad=1.0, w_pad=1.0)

    # 6) Export
    img_b64 = fig_to_base64(fig)
    plt.close(fig)
    return f"data:image/png;base64,{img_b64}"


def train_model(data: pd.DataFrame, features: List[str], target: str) -> str:
    print("[train_model] Preparing data...")

    X_raw = data[features]
    y = data[target]

    numeric_feats = X_raw.select_dtypes(include="number").columns.tolist()
    categorical_feats = [col for col in X_raw.columns if col not in numeric_feats]

    # Encode categorical features
    encoder = TargetEncoder(cols=categorical_feats)
    X_encoded = pd.concat([
        X_raw[numeric_feats],
        encoder.fit_transform(X_raw[categorical_feats], y)
    ], axis=1)

    # Save encoder
    joblib.dump(encoder, "models/target_encoder.joblib")
    joblib.dump(categorical_feats, "models/target_encoded_columns.joblib")

    # Train/test split
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=stratify)

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Run agents
    agents = [
        ("RandomForest", RandomForestAgent()),
        ("LogisticRegression", LogisticRegressionAgent()),
        ("DecisionTree", DecisionTreeAgent()),
        # ("SVM", SVMClassifierAgent()),  # Add back if needed
    ]

    best_model_path = None
    best_f1 = -1.0

    for name, agent in agents:
        try:
            print(f"[train_model] Running {name}")
            model, report = agent.train(X_train_resampled, y_train_resampled, X_test, y_test)
            f1 = report.get("macro avg", {}).get("f1-score", 0.0)
            path = f"models/{name}_model.joblib"
            joblib.dump(model, path)

            if f1 > best_f1:
                best_f1 = f1
                best_model_path = path

        except Exception as e:
            print(f"[train_model] {name} failed: {e}")

    if not best_model_path:
        raise RuntimeError("No model was successfully trained.")

    print(f"[train_model] ✅ Best model: {best_model_path} (F1={best_f1:.3f})")
    return best_model_path


def predict_severity(model_path: str, input_data: pd.DataFrame) -> List[Any]:

    print(f"[predict_severity] Loading model from: {model_path}")
    model = joblib.load(model_path)
    encoder = joblib.load("models/target_encoder.joblib")
    categorical_feats = joblib.load("models/target_encoded_columns.joblib")

    # Identify numeric columns
    numeric_feats = input_data.select_dtypes(include="number").columns.tolist()
    cat_feats = [col for col in input_data.columns if col not in numeric_feats]

    # Ensure columns match
    encoded_data = pd.concat([
        input_data[numeric_feats],
        encoder.transform(input_data[cat_feats])
    ], axis=1)

    # Make predictions
    predictions = model.predict(encoded_data)
    print("[predict_severity] ✅ Predictions complete.")
    return predictions.tolist()


def generate_plot(command: str, csv_path: str) -> str:
    df = pd.read_csv(csv_path)
    os.makedirs("static/reports", exist_ok=True)

    output_dir = "static/reports"
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "plot.png")

    if "correlation" in command:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix")
        plt.tight_layout()
    elif "severity distribution" in command or "hist" in command:
        plt.figure()
        df["severity"].value_counts().plot(kind="bar")
        plt.title("Severity Distribution")
    elif "bar" in command and "cause" in command:
        plt.figure()
        df["cause"].value_counts().plot(kind="bar")
        plt.title("Root Causes")
    else:
        raise ValueError("Unsupported plot type")

    plt.savefig(fig_path, dpi=150)
    plt.close()
    return "plot.png"  # ✅ Just return filename relative to static/reports
