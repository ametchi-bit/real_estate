import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set up the title and subheaders for the app
st.title("Rapport des données enregistré des caméras de surveillance de Sophonia")
st.subheader("Caméra D09 - Rue Principale Entrée")
st.subheader("Scénario: Sens entrée")
st.subheader("Catégorie: Barrière à sens unique")

# Load the data
entree_principal_entree_df = pd.read_csv("entree principle sens entree.csv")
entree_principal_sortie_df = pd.read_csv("entree principale sens sortie.csv")

# Drop unused columns
entree_principal_entree_df = entree_principal_entree_df.drop(
    columns=["Scénario", "Caméra", "Catégorie"]
)
entree_principal_sortie_df = entree_principal_sortie_df.drop(
    columns=["Scénario", "Caméra", "Catégorie"]
)

# Rename columns for comparison
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

# Select specific columns from each DataFrame for comparison
selected_columns_entree = entree_principal_entree_comparatif_df[
    ["Horodatage_in", "Type d'objet_in", "Nombre_in"]
]
selected_columns_sortie = entree_principal_sortie_comparatif_df[
    ["Horodatage_out", "Type d'objet_out", "Nombre_out"]
]

# ---------------------- Data Visualization -----------------------

# Sens entrée: Create donut chart
labels_entree = entree_principal_entree_df["Type d'objet"]
values_entree = entree_principal_entree_df["Nombre"]

piefig_entree = go.Figure(
    data=[go.Pie(labels=labels_entree, values=values_entree, hole=0.3)]
)
piefig_entree.update_layout(title_text="Nombre d'entrées par type d'objet")

# Convert Horodatage to datetime and set as index
entree_principal_entree_df["Horodatage"] = pd.to_datetime(
    entree_principal_entree_df["Horodatage"], format="%d/%m/%Y %H:%M:%S"
)
entree_principal_entree_df.set_index("Horodatage", inplace=True)

# Group by week and type of vehicle, and sum the entries
weekly_entree_df = (
    entree_principal_entree_df.groupby([pd.Grouper(freq="W"), "Type d'objet"])["Nombre"]
    .sum()
    .reset_index()
)

# Plot the bar chart for Sens Entrée
barfig_entree = px.bar(
    weekly_entree_df,
    x="Horodatage",
    y="Nombre",
    color="Type d'objet",
    title="Nombre d'entrées par type d'objet par semaine",
)

# Sens sortie: Create donut chart
labels_sortie = entree_principal_sortie_df["Type d'objet"]
values_sortie = entree_principal_sortie_df["Nombre"]

piefig_sortie = go.Figure(
    data=[go.Pie(labels=labels_sortie, values=values_sortie, hole=0.3)]
)
piefig_sortie.update_layout(title_text="Nombre de sorties par type d'objet")

# Convert Horodatage to datetime and set as index
entree_principal_sortie_df["Horodatage"] = pd.to_datetime(
    entree_principal_sortie_df["Horodatage"], format="%d/%m/%Y %H:%M:%S"
)
entree_principal_sortie_df.set_index("Horodatage", inplace=True)

# Group by week and type of vehicle, and sum the exits
weekly_sortie_df = (
    entree_principal_sortie_df.groupby([pd.Grouper(freq="W"), "Type d'objet"])["Nombre"]
    .sum()
    .reset_index()
)

# Plot the bar chart for Sens Sortie
barfig_sortie = px.bar(
    weekly_sortie_df,
    x="Horodatage",
    y="Nombre",
    color="Type d'objet",
    title="Nombre de sorties par type d'objet par semaine",
)

# --------------------- Data Comparison --------------------
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

# Extract day of the week and hour
combined_df["Jour de la semaine"] = combined_df["Horodatage"].dt.day_name()
combined_df["Heure"] = combined_df["Horodatage"].dt.hour

# Total volumes
total_in = combined_df["Nombre_in"].sum()
total_out = combined_df["Nombre_out"].sum()

# Group by Type d'objet and sum
vehicule_comparison = (
    combined_df.groupby("Type d'objet")[["Nombre_in", "Nombre_out"]].sum().reset_index()
)

# Create bar chart for comparison
fig_comparaison = px.bar(
    vehicule_comparison,
    x="Type d'objet",
    y=["Nombre_in", "Nombre_out"],
    barmode="group",
    title="Comparaison des volumes d'entrées et de sorties par type d'objet",
    labels={"value": "Nombre", "Type d'objet": "Type de véhicule"},
)

# Create the donut chart for Nombre_in
fig_in = go.Figure(
    data=[
        go.Pie(
            labels=vehicule_comparison["Type d'objet"],
            values=vehicule_comparison["Nombre_in"],
            hole=0.3,
        )
    ]
)
fig_in.update_layout(
    title_text="Nombre d'entrées par type d'objet",
    annotations=[dict(text="Entrées", x=0.5, y=0.5, font_size=20, showarrow=False)],
)

# Create the donut chart for Nombre_out
fig_out = go.Figure(
    data=[
        go.Pie(
            labels=vehicule_comparison["Type d'objet"],
            values=vehicule_comparison["Nombre_out"],
            hole=0.3,
        )
    ]
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
    title="Correlation entre le Nombre d'entrée et le Nombre de sortie",
    labels={"Nombre_in": "Nombre d'entrées", "Nombre_out": "Nombre de sorties"},
)

# Group by week and type of vehicle, and sum the entries and exits
weekly_combined_df = (
    combined_df.groupby([pd.Grouper(key="Horodatage", freq="W"), "Type d'objet"])
    .sum()
    .reset_index()
)

# Plot the line chart
fig_weekly = px.line(
    weekly_combined_df,
    x="Horodatage",
    y="Nombre_in",
    color="Type d'objet",
    title="Nombre d'entrées par type d'objet par semaine",
    labels={"Nombre_in": "Nombre d'entrées", "Horodatage": "Semaine"},
)
fig_weekly.add_traces(
    px.line(
        weekly_combined_df, x="Horodatage", y="Nombre_out", color="Type d'objet"
    ).data
)
fig_weekly.update_layout(yaxis_title="Nombre", xaxis_title="Semaine")


# Function to interpret correlation
def interpret_correlation(correlation_matrix):
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


# Create a heatmap of the correlation matrix
fig_corr_heatmap = px.imshow(
    correlation,
    text_auto=True,
    aspect="auto",
    title="Correlation Heatmap",
    labels={"color": "Correlation Coefficient"},
)
interpretation = interpret_correlation(correlation)

# Tab layout for the Streamlit app
tab1, tab2, tab3 = st.tabs(["Sens Entrée", "Sens Sortie", "Comparaison des données"])

with tab1:
    st.subheader("Extrait des donnée Sens Entrée")
    st.write(entree_principal_entree_df)
    st.subheader("Analyse descriptive")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(piefig_entree)
    with col2:
        st.plotly_chart(barfig_entree)

with tab2:
    st.header("Extrait des donnée Sens Sortie")
    st.write(entree_principal_sortie_df)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(piefig_sortie)
    with col2:
        st.plotly_chart(barfig_sortie)

with tab3:
    st.header("Données jointes")
    st.write(combined_df)
    st.subheader("Type de comparaison")
    st.text(
        f"1. Comparaison des volumes totaux\n"
        f"2. Comparaison par type d'objet\n"
        f"3. Comparaison par jour de la semaine\n"
        f"4. Analyse des correlations"
    )

    st.markdown(
        '<h2 class="centered-subheader">1. Comparaison des volumes totaux</h2>',
        unsafe_allow_html=True,
    )
    st.write(
        f"Le total des entrées est : {total_in}, le total des sorties est : {total_out}"
    )

    st.markdown(
        '<h2 class="centered-subheader">2. Comparaison par type d\'objet</h2>',
        unsafe_allow_html=True,
    )
    st.write(vehicule_comparison)
    st.plotly_chart(fig_comparaison)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_in)
    with col2:
        st.plotly_chart(fig_out)

    st.markdown(
        '<h2 class="centered-subheader">3. Comparaison par jour de la semaine</h2>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_weekly)

    st.markdown(
        '<h2 class="centered-subheader">4. Analyse des correlations</h2>',
        unsafe_allow_html=True,
    )
    st.write(correlation)
    st.plotly_chart(fig_corr)
    st.write(interpretation)


def main():
    st.sidebar.title("Barre d'outils")
    task = st.sidebar.radio("Tâche", ["Analyse descriptive", "Visualisation"])


main()
