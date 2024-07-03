import streamlit as st
import pandas as pd
import plotly.express as px
from plotly.offline import plot
import plotly.io as pio
import plotly.graph_objects as go
import datetime
import io
from util import visualize_and_interpret


st.title("Rapport des données enregistré des caméras de surveillance de Sophonia")
st.subheader("Caméra D09 - Rue Principale Enntrée")
st.subheader("Scénario: Sens entrée")
st.subheader("Catégorie: Barrière à sens unique")

# importing data for cleaning
entree_principal_entree_df = pd.read_csv(
    "entree principle sens entree.csv", delimiter=","
)
entree_principal_sortie_df = pd.read_csv(
    "entree principale sens sortie.csv", delimiter=","
)

# Dropping unused columns
entree_principal_entree_df = entree_principal_entree_df.drop(
    columns=["Scénario", "Caméra", "Catégorie"]
)
entree_principal_sortie_df = entree_principal_sortie_df.drop(
    columns=["Scénario", "Caméra", "Catégorie"]
)

# creating new dataframe to compare the in and out data

# first  rename the data
entree_principal_entree_comparatif_df = entree_principal_entree_df.rename(
    columns={
        "Horodatage": "Horodatage_in",
        "Type d'objet": "Type d'objet_in",
        "Nombre": "Nombre_in",
    }
)

entree_principal_sortie_comparatif_df = entree_principal_sortie_df.rename(
    columns={
        "Horodatage": "Horodatage_out",
        "Type d'objet": "Type d'objet_out",
        "Nombre": "Nombre_out",
    }
)

# Select specific columns from each DataFrame
selected_columns_entree = entree_principal_entree_comparatif_df[
    ["Horodatage_in", "Type d'objet_in", "Nombre_in"]
]
selected_columns_sortie = entree_principal_sortie_comparatif_df[
    ["Horodatage_out", "Type d'objet_out", "Nombre_out"]
]

# ---------------------- data visualisation -----------------------

# ----------- sens entree------------

# Créer un graphique circulaire en style donut
labels = entree_principal_entree_df["Type d'objet"]
values = entree_principal_entree_df["Nombre"]
week = entree_principal_entree_df["Horodatage"]

piefig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
piefig.update_layout(title_text="Nombre d'entrées par type d'objet")

# Convert Horodatage to datetime
entree_principal_entree_df["Horodatage"] = pd.to_datetime(
    entree_principal_entree_df["Horodatage"], format="%d/%m/%Y %H:%M:%S"
)

# Set Horodatage as index
entree_principal_entree_df.set_index("Horodatage", inplace=True)

# Group by week and type of vehicle, and sum the entries
weekly_df = (
    entree_principal_entree_df.groupby([pd.Grouper(freq="W"), "Type d'objet"])["Nombre"]
    .sum()
    .reset_index()
)

# Plot the bar chart
fig = px.bar(
    weekly_df,
    x="Horodatage",
    y="Nombre",
    color="Type d'objet",
    title="Nombre d'entrées par type d'objet par semaine",
)

# --------------------- sens sortie --------------------
# Créer un graphique circulaire en style donut
labels = entree_principal_sortie_df["Type d'objet"]
values = entree_principal_sortie_df["Nombre"]
week = entree_principal_sortie_df["Horodatage"]

piefig2 = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
piefig2.update_layout(title_text="Nombre d'entrées par type d'objet")

# Convert Horodatage to datetime
entree_principal_sortie_df["Horodatage"] = pd.to_datetime(
    entree_principal_sortie_df["Horodatage"], format="%d/%m/%Y %H:%M:%S"
)

# Set Horodatage as index
entree_principal_sortie_df.set_index("Horodatage", inplace=True)

# Group by week and type of vehicle, and sum the entries
weekly_df = (
    entree_principal_sortie_df.groupby([pd.Grouper(freq="W"), "Type d'objet"])["Nombre"]
    .sum()
    .reset_index()
)

# Plot the bar chart
fig2 = px.bar(
    weekly_df,
    x="Horodatage",
    y="Nombre",
    color="Type d'objet",
    title="Nombre d'entrées par type d'objet par semaine",
)
st.subheader("Analyse des donnees de comptage du trafic")
st.text(
    f"Selon les données fournies, il est perçu qu'elles représentent le nombre de véhicules de\n"
    f"personnes et tout autres franchissant un point d'entrée/sortie muni d'une barrière à sens unique. Les données\n"
    f"sont enregistrées par : horodatage, caméra, scénario, catégorie, type d'objet et nombre."
)

# creating tables
tab1, tab2, tab3 = st.tabs(["Sens Entrée", "Sens Sortie", "Comparaison des donnees"])

with tab1:

    st.subheader("Extrait des donnée Sens Entrée")
    st.write(entree_principal_entree_df)
    st.subheader("Analyse descriptive")

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(piefig)
    with col2:
        st.plotly_chart(fig)


with tab2:
    st.header("Extrait des donnée sens sortie")
    st.write(entree_principal_sortie_df)
    cols1, cols2 = st.columns(2)
    with cols1:
        st.plotly_chart(piefig2)
    with cols2:
        st.plotly_chart(fig2)

# ----------------------------- comparaison -----------------------------
# Combine the selected columns into a new DataFrame
combined_df = pd.concat([selected_columns_entree, selected_columns_sortie], axis=1)
combined_df = combined_df.drop(columns=["Type d'objet_out", "Horodatage_out"])

combined_df = combined_df.rename(
    columns={"Horodatage_in": "Horodatage", "Type d'objet_in": "Type d'objet"}
)
# Convert Horodatage to datetime
combined_df["Horodatage"] = pd.to_datetime(
    combined_df["Horodatage"], format="%d/%m/%Y %H:%M:%S"
)

# Extract day of the week
combined_df["Jour de la semaine"] = combined_df["Horodatage"].dt.day_name()

# Extract hour
combined_df["Heure"] = combined_df["Horodatage"].dt.hour
# Total volume
total_in = combined_df["Nombre_in"].sum()
total_out = combined_df["Nombre_out"].sum()

# Group by Type d'objet and sum
vehicule_comparison = combined_df.groupby("Type d'objet")[
    ["Nombre_in", "Nombre_out"]
].sum()


def interpret_correlation(correlation_matrix):
    """
    Generate an interpretation of the correlation matrix.

    Parameters:
    - correlation_matrix: pandas DataFrame containing the correlation matrix

    Returns:
    - interpretation: str containing the interpretation of the correlation matrix
    """
    interpretation = ""

    for row in correlation_matrix.index:
        for col in correlation_matrix.columns:
            if row != col:
                corr_value = correlation_matrix.loc[row, col]
                if corr_value == 1:
                    interpretation += f"The correlation between {row} and {col} is perfect positive (1). They move together exactly.\n"
                elif corr_value == -1:
                    interpretation += f"The correlation between {row} and {col} is perfect negative (-1). They move in exact opposite directions.\n"
                elif corr_value > 0.7:
                    interpretation += f"The correlation between {row} and {col} is strong positive ({corr_value:.2f}). They tend to increase together.\n"
                elif corr_value < -0.7:
                    interpretation += f"The correlation between {row} and {col} is strong negative ({corr_value:.2f}). One tends to increase as the other decreases.\n"
                elif corr_value > 0.3:
                    interpretation += f"The correlation between {row} and {col} is moderate positive ({corr_value:.2f}). They have a moderate tendency to increase together.\n"
                elif corr_value < -0.3:
                    interpretation += f"The correlation between {row} and {col} is moderate negative ({corr_value:.2f}). One tends to moderately increase as the other decreases.\n"
                elif corr_value > 0:
                    interpretation += f"The correlation between {row} and {col} is weak positive ({corr_value:.2f}). They have a weak tendency to increase together.\n"
                elif corr_value < 0:
                    interpretation += f"The correlation between {row} and {col} is weak negative ({corr_value:.2f}). One tends to weakly increase as the other decreases.\n"
                else:
                    interpretation += f"The correlation between {row} and {col} is zero ({corr_value:.2f}). They do not affect each other.\n"

    return interpretation


def visualize_and_interpret(combined_df):
    """
    Generate a heatmap of the correlation matrix and provide an interpretation.

    Parameters:
    - combined_df: pandas DataFrame containing the data

    Returns:
    - fig: plotly heatmap figure
    - interpretation: str containing the interpretation of the heatmap
    """
    # Filter numeric columns for correlation calculation
    numeric_df = combined_df.select_dtypes(include=[float, int])

    # Calculate correlation
    correlation_matrix = numeric_df.corr()

    # Create a heatmap
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap",
        labels={"color": "Correlation Coefficient"},
    )

    # Generate interpretation
    interpretation = interpret_correlation(correlation_matrix)

    return fig, interpretation


# Use the function to visualize and interpret
fig, interpretation = visualize_and_interpret(combined_df)

# Show plot
fig.show()

# Print interpretation
print(interpretation)

# Reset index for plotting
vehicule_comparison = vehicule_comparison.reset_index()

# Create bar chart
fig_comparaison = px.bar(
    vehicule_comparison,
    x="Type d'objet",
    y=["Nombre_in", "Nombre_out"],
    barmode="group",
    title="Comparaison des volumes d'entrées et de sorties par type d'objet",
    labels={"value": "Nombre", "Type d'objet": "Type de véhicule"},
)

# Prepare data for the donut chart
labels = vehicule_comparison.index
values_in = vehicule_comparison["Nombre_in"]
values_out = vehicule_comparison["Nombre_out"]

# Create the donut chart for Nombre_in
fig_in = go.Figure(
    data=[go.Pie(labels=labels, values=values_in, hole=0.3, name="Nombre d'entrées")]
)
fig_in.update_layout(
    title_text="Nombre d'entrées par type d'objet",
    annotations=[dict(text="Entrées", x=0.5, y=0.5, font_size=20, showarrow=False)],
)

# Create the donut chart for Nombre_out
fig_out = go.Figure(
    data=[go.Pie(labels=labels, values=values_out, hole=0.3, name="Nombre de sorties")]
)
fig_out.update_layout(
    title_text="Nombre de sorties par type d'objet",
    annotations=[dict(text="Sorties", x=0.5, y=0.5, font_size=20, showarrow=False)],
)

# Calculate correlation
correlation = combined_df[["Nombre_in", "Nombre_out"]].corr()

# Create scatter plot
fig_corr = px.scatter(
    combined_df,
    x="Nombre_in",
    y="Nombre_out",
    trendline="ols",
    title="Correlation entre le Nombre d'entree et le Nombre de sortie",
    labels={
        "Nombre_in": "Number of Entries (Nombre_in)",
        "Nombre_out": "Number of Exits (Nombre_out)",
    },
)

# Group by week and type of vehicle, and sum the entries and exits
weekly_df = (
    combined_df.groupby([pd.Grouper(key="Horodatage", freq="W"), "Type d'objet"])
    .sum()
    .reset_index()
)

# Plot the line chart
fig_weekly = px.line(
    weekly_df,
    x="Horodatage",
    y="Nombre_in",
    color="Type d'objet",
    title="Nombre d'entrées par type d'objet par semaine",
    labels={"Nombre_in": "Nombre d'entrées", "Horodatage": "Semaine"},
)

fig_weekly.add_trace(
    px.line(weekly_df, x="Horodatage", y="Nombre_out", color="Type d'objet").data[0]
)

fig_weekly.update_layout(yaxis_title="Nombre", xaxis_title="Semaine")

with tab3:
    st.header("Données jointes")
    st.write(combined_df)
    st.subheader("Type de comparaison")
    st.text(
        f"1. Comparaison des volumes totaux\n"
        f"2. Comparaison par type d'objet\n"
        f"3. Comparaison par jour de la semaine\n"
        f"4. Analyse des correlation"
    )
    # Custom CSS to center the subheader
    st.markdown(
        """
        <style>
        .centered-subheader {
            text-align: center;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Applying the custom CSS class to the subheader
    st.markdown(
        '<h2 class="centered-subheader">1. Comparaison des volumes totaux</h2>',
        unsafe_allow_html=True,
    )
    st.subheader("")
    st.write(f"Le total des entree est :{total_in}, le total des sortie est{total_out}")

    # Applying the custom CSS class to the subheader
    st.markdown(
        '<h2 class="centered-subheader">2. Comparaison par type d\'objet</h2>',
        unsafe_allow_html=True,
    )
    st.subheader("")
    st.write(vehicule_comparison)
    st.plotly_chart(fig_comparaison)
    col_figIn, col_figOut = st.columns(2)
    with col_figIn:
        st.plotly_chart(fig_in)
    with col_figOut:
        st.plotly_chart(fig_out)

    # Applying the custom CSS class to the subheader
    st.markdown(
        '<h2 class="centered-subheader">3. Comparaison par jour de la semaine</h2>',
        unsafe_allow_html=True,
    )
    st.subheader("")
    st.plotly_chart(fig_weekly)

    # Applying the custom CSS class to the subheader
    st.markdown(
        '<h2 class="centered-subheader">4. Analyse des correlations</h2>',
        unsafe_allow_html=True,
    )
    st.subheader("")
    st.write(correlation)
    st.plotly_chart(fig_corr)
    st.write(interpretation)


# def analyse_desc(entree_principal_entree_df, entree_principal_sortie_df, combined_df):

# def visualisation(entree_principal_entree_df, entree_principal_sortie_df, combined_df):


def main():
    st.sidebar.title("Barre d'outils")
    task = st.sidebar.radio("Tache", ["Analyse descriptive", "Visualisation"])
    # if task == "Analyse Descriptive":
    #   analyse_desc(entree_principal_entree_df)
    # else:
    #   visualisation()


main()
