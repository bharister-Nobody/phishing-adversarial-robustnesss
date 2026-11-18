
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def plot_attack_success_rate(df, title, filename):
    success_rate = df["result_type"].value_counts(normalize=True).reset_index()
    success_rate.columns = ["Result Type", "Proportion"]

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Result Type", y="Proportion", data=success_rate)
    plt.title(title)
    plt.ylabel("Proportion")
    plt.xlabel("Result Type")
    plt.ylim(0, 1) # Ensure y-axis is consistent for comparison
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_num_queries_distribution(df, title, filename):
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x="num_queries", hue="result_type", kde=True, bins=20)
    plt.title(title)
    plt.xlabel("Number of Queries")
    plt.ylabel("Count")
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")

def plot_scores_scatter(df, title, filename):
    successful_attacks = df[df["result_type"] == "Successful"].copy()
    if not successful_attacks.empty:
        successful_attacks["original_score_percent"] = successful_attacks["original_score"] * 100
        successful_attacks["perturbed_score_percent"] = successful_attacks["perturbed_score"] * 100

        fig = px.scatter(
            successful_attacks,
            x="original_score_percent",
            y="perturbed_score_percent",
            title=title,
            labels={
                "original_score_percent": "Original Score (%)",
                "perturbed_score_percent": "Perturbed Score (%)"
            },
            hover_data={
                "original_text": True,
                "perturbed_text": True,
                "original_score": ":.4f",
                "perturbed_score": ":.4f",
                "ground_truth_output": True,
                "original_output": True,
                "perturbed_output": True
            }
        )
        fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], mode='lines', name='Identity Line', line=dict(color='gray', dash='dash')))
        fig.write_html(filename)
        print(f"Saved {filename}")
    else:
        print(f"No successful attacks to visualize scores for {title}.")

if __name__ == "__main__":
    original_log = "original_model_attack_log.csv"
    robust_log = "robust_model_attack_log.csv"

    df_original = pd.read_csv(original_log)
    df_robust = pd.read_csv(robust_log)

    # Plot for Original Model
    print("Generating visualizations for Original Model...")
    plot_attack_success_rate(df_original, "Original Model Attack Success Rate", "original_attack_success_rate.png")
    plot_num_queries_distribution(df_original, "Original Model: Distribution of Queries", "original_num_queries_distribution.png")
    plot_scores_scatter(df_original, "Original Model: Original vs. Perturbed Scores (Successful Attacks)", "original_scores_successful_attacks.html")

    # Plot for Robust Model
    print("\nGenerating visualizations for Robust Model...")
    plot_attack_success_rate(df_robust, "Robust Model Attack Success Rate", "robust_attack_success_rate.png")
    plot_num_queries_distribution(df_robust, "Robust Model: Distribution of Queries", "robust_num_queries_distribution.png")
    plot_scores_scatter(df_robust, "Robust Model: Original vs. Perturbed Scores (Successful Attacks)", "robust_scores_successful_attacks.html")

    # Comparative Analysis (Example: Bar chart for success rates)
    original_success_rate = df_original["result_type"].value_counts(normalize=True).get("Successful", 0)
    robust_success_rate = df_robust["result_type"].value_counts(normalize=True).get("Successful", 0)

    comparison_df = pd.DataFrame({
        "Model": ["Original", "Robust"],
        "Successful Attack Rate": [original_success_rate, robust_success_rate]
    })

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Model", y="Successful Attack Rate", data=comparison_df)
    plt.title("Comparison of Successful Attack Rates: Original vs. Robust Model")
    plt.ylabel("Successful Attack Rate")
    plt.ylim(0, 1)
    plt.savefig("comparison_attack_success_rate.png")
    plt.close()
    print("Saved comparison_attack_success_rate.png")

    # You can add more comparative plots here if needed, e.g., side-by-side histograms or combined scatter plots


