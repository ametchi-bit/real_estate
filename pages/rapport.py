import streamlit as st
from generate_report_global import generate_report
from generate_report_perCam import generate_report_cam
import os
import pandas as pd
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.io as pio
from plotly.subplots import make_subplots
import uuid
from datetime import datetime
from io import BytesIO
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
    repartition_viz,
    count_by_type_objet
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
     # Generate figures for visualizations
    
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
        '<h4 style="text-align: left; color: #000000;">Aperçu de l\'ensemble des donnees</h4>',
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
    st.subheader("Statistiques Récapitulatives")
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
    rep_cam_fig, rep_scen_fig, fig_categorie,image_stream_cam,image_stream_cat, image_stream_scen = repartition_viz(symphonia_data)

    st.markdown(
        "### Répartition des donnees en fonction des cameras, Scenario et categorie "
    )
    st.markdown(
        '<h3 style="text-align: center;">Caméra</h3>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(rep_cam_fig)
    st.markdown(
        '<h3 style="text-align: center;">Scénario</h3>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(rep_scen_fig)
    st.markdown(
        '<h3 style="text-align: center;">Catégorie</h3>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig_categorie)
    
    # comparaison des donnees 
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">comparaison des donnees</h4>',
        unsafe_allow_html=True,
    )
    comparaison_fig, image_stream_comparaison = plot_comparaison_type_objet_real_estate(symphonia_data)
    st.plotly_chart(comparaison_fig)

    # Trends and patterns
    st.subheader("Tendance et modele")
    mwd_fig, image_stream_mwd = plot_weekly_daily_patterns_real_estate(symphonia_data, "Horodatage", "Nombre")
    st.plotly_chart(mwd_fig)

    # Anomalies and Insight
    st.subheader("Detection d'anomalies")
    st.write("Aucune anomalies detectee")

    # Conclusion
    st.subheader("Conclusion")
    st.write("Resune")
    conclusion_text = st.text_area("Entrer ou Modifier le text", "")
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">Recommendation</h5>',
        unsafe_allow_html=True,
    )
    recommendation_text = st.text_area("Modifier le text", "")

    # Appendices
    st.subheader("Appendices")
    st.write("glossaire")
    if st.button("Générer rapport"):
        pdf_data = generate_report(
            title=report_title,
            subtitle="Surveillance Data Analysis Report",
            intro_title="INTRODUCTION",
            title_objectif="Objectif",
            intro_objectif=report_objectif,
            title_scope="Scope",
            text_scope=report_scope,
            title_data_desc="Description des donnees",
            sub_title_data_text="Aperçu de l'ensemble des donnees",
            data_overview_text=report_data_overview,
            sub_title_data_desc_col="Colones des donnees",
            col_data_desc_col=report_col_desc,
            data_cleaning_title="Nettoyage et Preparation des donnee",
            missing_val_title="Valeurs Manquantes",
            missing_val_content=report_missing_val,
            transformation_data_title="Tansformation des donnees.",
            transformation_data_content=report_data_transform,
            desc_stat_title="Statistiques Récapitulatives",
            desc_stat_content=report_desc_info,
            data_viz = "Visualisation des donnees",
            conclusion_title="Conclusion",
            conclusion_subtitl_r="Resume",
            resume_content=conclusion_text,
            recomm_title="Recommendation",
            recomm_content=recommendation_text,
        )
        if pdf_data:
            st.download_button(
                label="Télécharger le fichier PDF",
                data=pdf_data,
                #file_name="rdp.pdf",
                mime="application/pdf",
            )
        else:
            st.error("Erreur lors de la génération du rapport PDF.")
        st.success("Rapport généré avec succès!")


# Define function to generate the camera-specific report
def rapport_par_camera(selected_df):
    # Generate figures for visualizations
    
    st.subheader("Titre du rapport")
    report_title = st.text_input(
        "Entrez le titre du rapport", f"Rapport des donnée de surveillance de la caméra:"
    )
    st.subheader("Introduction")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">Objectif</h4>',
        unsafe_allow_html=True,
    )
    report_objectif = st.text_area(
        "Entrez ou modifier l'objectif",
        "Ce rapport vise à analyser les données de circulation captées par la caméra de vidéosurveillance installées à l'entrée principale de la propriété immobilière.",
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
        '<h4 style="text-align: left; color: #000000;">Aperçu de l\'ensemble des donnees</h4>',
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
    st.subheader("Statistiques Récapitulatives")
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
    
    counts = count_by_type_objet(selected_df)
    
    # Accessing the counts
    bus_count = counts['bus']
    moto_count = counts['moto']
    personne_count = counts['personne']
    vehicule_intermediaire_count = counts['véhicule intermédiaire']
    vehicule_leger_count = counts['véhicule léger']
    velo_count = counts['vélo']

    t_o_col1, t_o_col2, t_o_col3 = st.columns(3)
    with t_o_col1:
        st.write(f"Bus: {bus_count}")
    with t_o_col2:
        st.write(f"Moto: {moto_count}")
    with t_o_col3:
        st.write(f"Personne: {personne_count}")
    with t_o_col1:  
        st.write(f"Véhicule Intermédiaire: {vehicule_intermediaire_count}")
    with t_o_col2:
        st.write(f"Véhicule Léger: {vehicule_leger_count}")
    with t_o_col3:
        st.write(f"Vélo: {velo_count}")

    #st.dataframe(distribution_real_estate(selected_df, "Type d'objet", "Nombre"))

    if selected_df is not None:
        distribution_fig = px.histogram(
            selected_df,
            x="Type d'objet",
            y="Nombre",
            title="Distribution des données",
        )
        st.plotly_chart(distribution_fig)

    # Data Visualizations
    st.subheader("Visualisation des donnees")
    

    
    # comparaison des donnees 
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">comparaison des donnees</h4>',
        unsafe_allow_html=True,
    )
    comparaison_fig, image_stream_comparaison = plot_comparaison_type_objet_real_estate(selected_df)
    st.plotly_chart(comparaison_fig)

    # Trends and patterns
    st.subheader("Tendance et modele")
    mwd_fig, image_stream_mwd = plot_weekly_daily_patterns_real_estate(selected_df, "Horodatage", "Nombre")
    st.plotly_chart(mwd_fig)

    # Anomalies and Insight
    st.subheader("Detection d'anomalies")
    st.write("Aucune anomalies detectee")

    # Conclusion
    st.subheader("Conclusion")
    st.write("Resune")
    conclusion_text = st.text_area("Entrer ou Modifier le text", "")
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">Recommendation</h5>',
        unsafe_allow_html=True,
    )
    recommendation_text = st.text_area("Modifier le text", "")

    # Appendices
    st.subheader("Appendices")
    st.write("glossaire")
    if st.button("Générer rapport"):
        pdf_data = generate_report(
            title=report_title,
            subtitle="Surveillance Data Analysis Report",
            intro_title="INTRODUCTION",
            title_objectif="Objectif",
            intro_objectif=report_objectif,
            title_scope="Scope",
            text_scope=report_scope,
            title_data_desc="Description des donnees",
            sub_title_data_text="Aperçu de l'ensemble des donnees",
            data_overview_text=report_data_overview,
            sub_title_data_desc_col="Colones des donnees",
            col_data_desc_col=report_col_desc,
            data_cleaning_title="Nettoyage et Preparation des donnee",
            missing_val_title="Valeurs Manquantes",
            missing_val_content=report_missing_val,
            transformation_data_title="Tansformation des donnees.",
            transformation_data_content=report_data_transform,
            desc_stat_title="Statistiques Récapitulatives",
            desc_stat_content=report_desc_info,
            data_viz = "Visualisation des donnees",
            conclusion_title="Conclusion",
            conclusion_subtitl_r="Resume",
            resume_content=conclusion_text,
            recomm_title="Recommendation",
            recomm_content=recommendation_text,
        )
        if pdf_data:
            st.download_button(
                label="Télécharger le fichier PDF",
                data=pdf_data,
                #file_name="rdp.pdf",
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
