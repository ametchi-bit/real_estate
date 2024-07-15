import streamlit as st
from generate_report import generate_report
import os
import pandas as pd
import base64
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from util import (
    symphonia_data,
    visualization_pie_repartition,
    summary_desc_symphonia,
    display_column_descriptions,
    explain_summary_desc,
    distribution_real_estate,
    plot_real_estate_histogram,
    comparaison_type_objet_real_estate,
    plot_comparaison_type_objet_real_estate,
    calculate_monthly_weekly_daily_patterns_real_estate,
    plot_weekly_daily_patterns_real_estate,
)


# Load datasets
def load_data(folder_path):
    files = [
        f
        for f in os.listdir(folder_path)
        if os.path.isfile(os.path.join(folder_path, f))
    ]
    dataframes = {file: pd.read_csv(os.path.join(folder_path, file)) for file in files}
    return dataframes


folder_path = "dataframe"
dataframes = load_data(folder_path)


def handle_missing_val_real_estate(data):
    """
    Filter out rows where 'Nombre' is zero.

    Args:
    - data (pd.DataFrame): Input DataFrame to filter.

    Returns:
    - filtered_data (pd.DataFrame): Data with rows where 'Nombre' is zero removed.
    """
    filtered_data = data[data["Nombre"] != 0]  # Filter rows where 'Nombre' is not zero
    return filtered_data


# Page title
st.markdown(
    '<div class="custom-container">'
    '<h1 style="text-align: center; color: #FF4B4B;">Rapport d\'analyse des données de surveillance</h1>'
    "</div>",
    unsafe_allow_html=True,
)
st.divider()


# Define function to generate the global report
def rapport_global(symphonia_data):
    st.subheader("Titre du rapport")
    report_title = st.text_input(
        "Entrez le titre du rapport", "Rapport Global des Données de Surveillance"
    )
    st.subheader("Introduction")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Objectif</h4>',
        unsafe_allow_html=True,
    )
    report_objectif = st.text_area(
        "Entrez ou modifier l'objectif",
        "Ce rapport vise à analyser les données de circulation captées par les caméras de vidéosurveillance installées à l'entrée principale de la propriété immobilière.",
    )

    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Scope</h4>',
        unsafe_allow_html=True,
    )
    report_scope = st.text_area(
        "Entrez ou modifier le Scope",
        "Les données comprennent des enregistrements de trafic de janvier à juin 2024, capturant divers types d'objets tels que des véhicules et des piétons.",
    )

    # Data description
    st.subheader("Description des donnees")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Appercue de l\'ensemble des donnees</h4>',
        unsafe_allow_html=True,
    )
    report_data_overview = st.text_area(
        "Entrez ou modifier le text",
        "L'ensemble de données contient 10 000 enregistrements de janvier à juin 2024. Il comprend des informations sur les types d'objets détectés, les horodatages et les décomptes.",
    )
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">colonnes des donnees</h4>',
        unsafe_allow_html=True,
    )
    report_col_desc = display_column_descriptions(symphonia_data)
    if report_col_desc is None:
        report_col_desc = "No column descriptions available."

    # Data cleaning and Preparation
    st.subheader("Nettoyage et preparation des donnees")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Valeurs Manquantes</h4>',
        unsafe_allow_html=True,
    )
    report_missing_val = st.text_area(
        "Entrez ou modifier le text",
        "Les valeurs manquantes ont été traitées en supprimant les enregistrements incomplets.",
    )
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Transformation des donnees</h4>',
        unsafe_allow_html=True,
    )
    report_data_transform = st.text_area(
        "Entrez ou modifier le text",
        "Les horodatages ont été convertis au format datetime et les colonnes non pertinentes ont été supprimées.",
    )

    # Descriptive Statistics
    st.subheader("Summary Statistics")
    report_desc_info = st.dataframe(summary_desc_symphonia(symphonia_data))
    desc_transposed = summary_desc_symphonia(symphonia_data[["Nombre"]])
    explanations = explain_summary_desc(desc_transposed)
    st.write("Explanations of the Summary Statistics")
    for column, stats in explanations.items():
        st.markdown(f"### {column}")
        for stat, explanation in stats.items():
            st.write(f"- **{stat}**: {explanation}")

    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Distribution des donnees</h4>',
        unsafe_allow_html=True,
    )

    st.dataframe(distribution_real_estate(symphonia_data, "Type d'objet", "Nombre"))

    if symphonia_data is not None:
        distribution_fig = px.histogram(
            symphonia_data,
            x="Type d'objet",
            y="Nombre",
            title="Distribution des données",
        )
        st.plotly_chart(distribution_fig)

    # Data Visualizations
    st.subheader("Visualisation des donnees")
    # Group and aggregate the data
    grouped_data = (
        symphonia_data.groupby(["Caméra", "Scénario", "Catégorie", "Type d'objet"])
        .agg({"Nombre": "sum"})
        .reset_index()
    )

    # Extract relevant statistics for interpretation

    camera_counts = grouped_data.groupby("Caméra")["Nombre"].sum()
    scenario_counts = grouped_data.groupby("Scénario")["Nombre"].sum()
    category_counts = grouped_data.groupby("Catégorie")["Nombre"].sum()

    st.markdown(
        "### Répartition des donnees en fonction des cameras, Scenario et categorie "
    )

    st.markdown(
        '<h3 style="text-align: center;">Caméra</h3>',
        unsafe_allow_html=True,
    )
    camera_data_list = []
    for camera, count in camera_counts.items():
        # st.markdown(f"- **{camera}**: {count} événements")
        camera_data_list.append({"Caméra": camera, "événements": count})
    rep_cam = pd.DataFrame(camera_data_list)

    # Creating the pie chart
    rep_cam_fig = px.pie(
        rep_cam,
        values="événements",
        names="Caméra",
        color_discrete_sequence=px.colors.sequential.RdBu,
        width=600,  # Set the width of the plot
        height=600,
    )
    st.plotly_chart(rep_cam_fig)

    st.markdown(
        '<h3 style="text-align: center;">Scénario</h3>',
        unsafe_allow_html=True,
    )
    scenario_data_list = []
    for scenario, count in scenario_counts.items():
        # st.markdown(f"- **{scenario}**: {count} événements")
        scenario_data_list.append({"Scénario": scenario, "événements": count})
    rep_scen = pd.DataFrame(scenario_data_list)

    # Creating the pie chart
    rep_scen_fig = px.pie(
        rep_scen,
        values="événements",
        names="Scénario",
        color_discrete_sequence=px.colors.sequential.RdBu,
        width=600,  # Set the width of the plot
        height=600,
    )
    st.plotly_chart(rep_scen_fig)

    st.markdown(
        '<h3 style="text-align: center;">Catégorie</h3>',
        unsafe_allow_html=True,
    )
    categorie_data_list = []
    for category, count in category_counts.items():
        # st.markdown(f"- **{category}**: {count} événements")
        categorie_data_list.append({"Catégorie": category, "événements": count})
    rep_cat = pd.DataFrame(categorie_data_list)

    # Creating the pie chart
    fig = px.pie(
        rep_cat,
        values="événements",
        names="Catégorie",
        color_discrete_sequence=px.colors.sequential.RdBu,
        width=600,  # Set the width of the plot
        height=600,
    )
    st.plotly_chart(fig)
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">comparaison des donnees</h4>',
        unsafe_allow_html=True,
    )
    symphonia_data_filtered = handle_missing_val_real_estate(symphonia_data)
    vehicule_comparison = (
        symphonia_data_filtered.groupby(["Type d'objet", "Scénario"])["Nombre"]
        .sum()
        .reset_index()
    )

    comparaison_fig = px.bar(
        vehicule_comparison,
        x="Scénario",
        y="Nombre",
        color="Type d'objet",
        barmode="group",
        title="Comparaison par type d'objet et scénario",
        labels={
            "Nombre": "Nombre",
            "Scénario": "Scénario",
            "Type d'objet": "Type d'objet",
        },
    )

    st.plotly_chart(comparaison_fig)

    # Trends and patterns
    st.subheader("Tendance et modele")
    # convert to proper datetime
    symphonia_data["Horodatage"] = pd.to_datetime(
        symphonia_data["Horodatage"], format="%d/%m/%Y %H:%M:%S"
    )
    symphonia_data["Month"] = symphonia_data["Horodatage"].dt.strftime("%Y-%m")

    symphonia_data["Horodatage"] = pd.to_datetime(symphonia_data["Horodatage"])
    monthly_pattern = symphonia_data.resample("ME", on="Horodatage")["Nombre"].sum()
    weekly_pattern = symphonia_data.resample("W", on="Horodatage")["Nombre"].sum()
    daily_pattern = symphonia_data.resample("D", on="Horodatage")["Nombre"].sum()

    monthly_pattern, weekly_pattern, daily_pattern = (
        calculate_monthly_weekly_daily_patterns_real_estate(
            symphonia_data, "Horodatage", "Nombre"
        )
    )

    mwd_fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=("Modèle  Mensuel", "Modèle Hebdomadaire", "Modèle quotidien"),
    )

    mwd_fig.add_trace(
        go.Scatter(
            x=monthly_pattern.index, y=monthly_pattern, mode="lines", name="Mois"
        ),
        row=1,
        col=1,
    )

    mwd_fig.add_trace(
        go.Scatter(
            x=weekly_pattern.index, y=weekly_pattern, mode="lines", name="Semaine"
        ),
        row=2,
        col=1,
    )

    mwd_fig.add_trace(
        go.Scatter(x=daily_pattern.index, y=daily_pattern, mode="lines", name="Jour"),
        row=3,
        col=1,
    )

    mwd_fig.update_layout(
        title_text="Modèle Mensuel, Hebdomadaire, et quotidien",
        height=800,
        showlegend=True,
    )

    st.plotly_chart(mwd_fig)

    # Anomalies and Insight
    st.subheader("Detection d'anomalies")
    st.write("Aucune anomalies detectee")

    # Conclusion
    st.subheader("Conclusion")
    st.text_input("Entrer ou Modifier le text")
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">Recommendation</h5>',
        unsafe_allow_html=True,
    )
    st.text_area("Entrer ou Modifier le text")

    # Appendices
    st.subheader("Appendices")
    st.write("glossaire")

    if st.button("Générer rapport"):
        pdf_data = generate_report(
            title=report_title,
            subtitle="Surveillance Data Analysis Report",
            objective=report_objectif,
            scope=report_scope,
            overview=report_data_overview,
            columns=report_col_desc,
            missing_values=report_missing_val,
            transformations=report_data_transform,
            summary_stats=report_desc_info,
            data_distribution=distribution_fig.to_image(
                format="png", width=800, height=600, scale=2.0
            ),
            traffic_volume_cam=rep_cam_fig.to_image(
                format="png", width=800, height=600, scale=2.0
            ),
            traffic_volume_scen=rep_scen_fig.to_image(
                format="png", width=800, height=600, scale=2.0
            ),
            scenarios=comparaison_fig.to_image(
                format="png", width=800, height=600, scale=2.0
            ),
            monthly_weekly_daily_patterns=mwd_fig.to_image(
                format="png", width=800, height=600, scale=2.0
            ),
            anomaly_detection="Aucune anomalies detectee",
            key_insights="",
            summary="",
            recommendations="",
            appendices="glossaire",
        )
        if pdf_data:
            st.download_button(
                label="Télécharger le fichier PDF",
                data=pdf_data,
                file_name="rapport.pdf",
                mime="application/pdf",
            )
        else:
            st.error("Erreur lors de la génération du rapport PDF.")
        st.success("Rapport généré avec succès!")


# Define function to generate the camera-specific report
def rapport_par_camera(selected_df):
    st.subheader("Titre du rapport")
    report_title = st.text_input(
        "Entrez le titre du rapport", "Rapport Global des Données de Surveillance"
    )
    st.subheader("Introduction")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Objectif</h4>',
        unsafe_allow_html=True,
    )
    report_objectif = st.text_area(
        "Entrez ou modifier l'objectif",
        "Ce rapport vise à analyser les données de circulation captées par les caméras de vidéosurveillance installées à l'entrée principale de la propriété immobilière.",
    )

    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Scope</h4>',
        unsafe_allow_html=True,
    )
    report_scope = st.text_area(
        "Entrez ou modifier le Scope",
        "Les données comprennent des enregistrements de trafic de janvier à juin 2024, capturant divers types d'objets tels que des véhicules et des piétons.",
    )

    # Data description
    st.subheader("Description des donnees")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Appercue de l\'ensemble des donnees</h4>',
        unsafe_allow_html=True,
    )
    report_data_overview = st.text_area(
        "Entrez ou modifier le text",
        "L'ensemble de données contient 10 000 enregistrements de janvier à juin 2024. Il comprend des informations sur les types d'objets détectés, les horodatages et les décomptes.",
    )
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">colonnes des donnees</h4>',
        unsafe_allow_html=True,
    )
    report_col_desc = display_column_descriptions(selected_df)
    if report_col_desc is None:
        report_col_desc = "No column descriptions available."

    # Data cleaning and Preparation
    st.subheader("Nettoyage et preparation des donnees")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Valeurs Manquantes</h4>',
        unsafe_allow_html=True,
    )
    report_missing_val = st.text_area(
        "Entrez ou modifier le text",
        "Les valeurs manquantes ont été traitées en supprimant les enregistrements incomplets.",
    )
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Transformation des donnees</h4>',
        unsafe_allow_html=True,
    )
    report_data_transform = st.text_area(
        "Entrez ou modifier le text",
        "Les horodatages ont été convertis au format datetime et les colonnes non pertinentes ont été supprimées.",
    )

    # Descriptive Statistics
    st.subheader("Summary Statistics")
    report_desc_info = st.dataframe(summary_desc_symphonia(selected_df))
    desc_transposed = summary_desc_symphonia(selected_df[["Nombre"]])
    explanations = explain_summary_desc(desc_transposed)
    st.write("Explanations of the Summary Statistics")
    for column, stats in explanations.items():
        st.markdown(f"### {column}")
        for stat, explanation in stats.items():
            st.write(f"- **{stat}**: {explanation}")

    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Distribution des donnees</h4>',
        unsafe_allow_html=True,
    )
    st.dataframe(distribution_real_estate(selected_df, "Type d'objet", "Nombre"))
    st.write(plot_real_estate_histogram(selected_df, "Type d'objet", "Nombre"))

    # Data Visualizations
    st.subheader("Visualisation des donnees")
    st.write(visualization_pie_repartition(selected_df))
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">comparaison des donnees</h4>',
        unsafe_allow_html=True,
    )
    st.dataframe(comparaison_type_objet_real_estate(selected_df))
    st.write(plot_comparaison_type_objet_real_estate(selected_df))

    # Trends and patterns
    st.subheader("Tendance et modele")
    st.write(
        plot_weekly_daily_patterns_real_estate(selected_df, "Horodatage", "Nombre")
    )

    # Anomalies and Insight
    st.subheader("Detection d'anomalies")
    st.write("Aucune anomalies detectee")

    # Conclusion
    st.subheader("Conclusion")
    st.text_input("Entrer ou Modifier le text")
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">Recommendation</h5>',
        unsafe_allow_html=True,
    )
    st.text_area("Entrer ou Modifier le text")

    # Appendices
    st.subheader("Appendices")
    st.write("glossaire")
    if st.button("Générer"):
        pdf_data = generate_report(
            title=report_title,
            subtitle="Surveillance Data Analysis Report",
            objective=report_objectif,
            scope=report_scope,
            overview=report_data_overview,
            columns=report_col_desc,
            missing_values=report_missing_val,
            transformations=report_data_transform,
            summary_stats=report_desc_info,
            data_distribution=distribution_real_estate(
                selected_df, "Type d'objet", "Nombre"
            ),
            traffic_volume=visualization_pie_repartition(selected_df),
            scenarios=comparaison_type_objet_real_estate(selected_df),
            monthly_trend=plot_weekly_daily_patterns_real_estate(
                selected_df, "Horodatage", "Nombre"
            ),
            weekly_daily_patterns=plot_weekly_daily_patterns_real_estate(
                selected_df, "Horodatage", "Nombre"
            ),
            anomaly_detection="Aucune anomalies detectee",
            key_insights="",
            summary="",
            recommendations="",
            appendices="glossaire",
        )
        if pdf_data:
            st.download_button(
                label="Télécharger le fichier PDF",
                data=pdf_data,
                file_name="rapport.pdf",
                mime="application/pdf",
            )
        else:
            st.error("Erreur lors de la génération du rapport PDF.")
    st.success("Rapport généré avec succès!")


def main():
    # Choose type of report
    st.sidebar.title("Type de rapport")
    report_type = st.sidebar.selectbox(
        "Sélectionnez le type de rapport", ["Rapport Global", "Rapport par Caméra"]
    )

    # Generate report based on selection
    if report_type == "Rapport Global":
        rapport_global(symphonia_data)
    else:
        camera_choice = st.sidebar.selectbox(
            "Sélectionnez une caméra", dataframes.keys()
        )
        selected_df = dataframes[camera_choice]
        rapport_par_camera(selected_df)


if __name__ == "__main__":
    main()
