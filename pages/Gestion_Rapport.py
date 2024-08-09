import streamlit as st
from generate_report_global import generate_report
from generate_report_perCam import generate_report_cam
import os
import pandas as pd
from docx import Document
from PyPDF2 import PdfReader
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
    analyse_comparative_par_camera,
    analyse_correlation,
    analyze_anomalies,
    analyze_scenario_variability,
    correlation_analysis,
    create_pivot_table,
    diagrammes_de_dispersion,
    ecart_type_intercameras,
    main_analysis,
    main_analysis_per_cam,
    preprocess_data,
    segmentation_grouping,
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

# Function to load files from a folder
def load_files(folder_path2):
    return [
        f for f in os.listdir(folder_path2)
        if os.path.isfile(os.path.join(folder_path2, f)) and (f.endswith('.docx') or f.endswith('.pdf'))
    ]

# Folder containing generated files
folder_path2 = "File Generated"
files = load_files(folder_path2)


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
        "Entrer le titre du rapport", "Rapport d'analyse des Données de VideoSurveillance"
    )
    st.subheader("Sous-titre du rapport")
    report_subtitle= st.text_input(
        "Entrer le sous-titre du rapport", "Analyse de Données de VideoSurveillance de la cité Symphonia"
    )
    st.subheader("I. Introduction")
    report_brief_overview = st.text_area("Entrer ou modifier le brief ", "Ce rapport d'analyse se concentre sur l'examen des données de vidéosurveillance de Symphonia, une propriété immobilière située à Abidjan, Cocody. Les données ont été recueillies grâce à l'application Videtics, qui exploite des technologies de vision par ordinateur pour surveiller efficacement la propriété. Videtics permet de positionner des lignes de comptage pour obtenir un décompte précis des différents types d'objets franchissant ces lignes, d'identifier et de distinguer divers types d'objets tels que les personnes, les motos, les véhicules légers et intermédiaires, les bus et les vélos, et de détecter les intrusions pour alerter sur des comportements suspects.")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">1. Objectif</h4>',
        unsafe_allow_html=True,
    )
    report_objectif = st.text_area(
        "Entrer ou modifier l'objectif",
        "L'objectif de ce rapport global est d'offrir une vue d'ensemble des tendances, des schémas et des anomalies détectées à partir des données collectées. En fournissant une analyse complète, nous visons à identifier des aperçus pertinents qui peuvent contribuer à la gestion efficace et à la sécurité renforcée de la propriété Symphonia. Ce rapport servira de fondation pour la prise de décisions informées, aidant ainsi à optimiser les ressources et à améliorer la sécurité et l'efficacité opérationnelle de Symphonia.",
    )

    st.markdown(
        '<h4 style="text-align: left; color: #000000;">2. Scope</h4>',
        unsafe_allow_html=True,
    )
    report_scope = st.text_area(
        "Entrer ou modifier le Scope",
        "Le scope de ce rapport inclut l'analyse des données de vidéosurveillance collectées sur une période définie, couvrant les différents types d'objets détectés, les fréquences de passage, et les événements d'intrusion signalés. Nous examinerons les données à travers plusieurs dimensions, notamment les répartitions mensuelles et hebdomadaires des différents types d'objets, les schémas de mouvement, et les anomalies détectées. L'analyse se concentrera également sur les tendances générales et les comportements inhabituels observés, fournissant ainsi des recommandations basées sur les aperçus dérivés des données. Ce rapport global couvre l'ensemble de la propriété Symphonia, offrant une perspective complète sur la sécurité et la gestion des ressources.",
    )

    # Data description
    st.subheader("II. Description des Données")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">1. Aperçu de l\'ensemble des données</h4>',
        unsafe_allow_html=True,
    )
    report_data_overview = st.text_area(
        "Entrer ou modifier le text",
        "Les données utilisées dans cette analyse proviennent de l'application Videtics, qui a surveillé en continu la propriété Symphonia située à Abidjan, Cocody. Les enregistrements couvrent une période spécifique et incluent des informations détaillées sur les divers objets détectés par les systèmes de vision par ordinateur. Cet ensemble de données est riche en informations, permettant une analyse approfondie des tendances et des comportements observés sur la propriété.",
    )
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">2. Description des colonnes des données</h4>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    - **Horodatage** : Cette colonne indique la date et l'heure précises de chaque événement enregistré. Elle est essentielle pour analyser les tendances temporelles et les schémas de comportement.
    - **Caméra** : Identifie la caméra spécifique ayant capturé l'événement. Cela permet de localiser géographiquement les événements et de comprendre les zones de la propriété où l'activité est la plus dense.
    - **Scénario** : Décrit le contexte ou la situation dans laquelle l'événement s'est produit, aidant ainsi à catégoriser les différentes situations de surveillance.
    - **Catégorie** : Classe les événements en différentes catégories, facilitant ainsi une analyse segmentée des données.
    - **Type d'objet** : Indique le type d'objet détecté, par exemple, personne, moto, véhicule léger, etc. Cette colonne est cruciale pour comprendre la distribution des différents objets sur la propriété.
    - **Nombre** : Quantifie le nombre d'objets détectés pour chaque enregistrement, permettant une analyse quantitative des données.
    """)


    st.markdown(
        '<h4 style="text-align: left; color: #000000;">3.Définition des types d\'objets</h4>',
        unsafe_allow_html=True,
    )
    intro_def_TO= st.write("Les types d'objets détectés et enregistrés par le système Videtics sont variés, couvrant les principaux éléments susceptibles de se déplacer dans la propriété Symphonia :")
    st.markdown("""
    - **Bus** : Véhiculent des passagers en grands nombres et nécessitent une surveillance particulière en raison de leur taille et de leur impact potentiel sur la circulation.
    - **Moto** : Inclut toutes les formes de motocyclettes, souvent plus rapides et plus difficiles à détecter que les véhicules plus grands.
    - **Personne** : Comprend tous les piétons, offrant des insights sur les mouvements des résidents, visiteurs et personnels.
    - **Véhicule intermédiaire** : Réfère aux véhicules de taille moyenne, tels que les camionnettes, qui jouent un rôle clé dans la logistique et le transport.
    - **Véhicule léger** : Inclut les voitures personnelles et autres véhicules de petite taille, couramment utilisés par les résidents et visiteurs.
    - **Vélo** : Inclut toutes les formes de bicyclettes, de plus en plus courantes dans les zones résidentielles et nécessitant une surveillance pour la sécurité.
    - **Poid lourd** : Véhicule de grande taille destiné au transport de marchandises, caractérisé par une capacité de charge élevée.
    """)
    
    # Data cleaning and Preparation
    st.subheader("III. Nettoyage et Préparation des Données")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">1. Gestion des Valeurs Manquantes</h4>',
        unsafe_allow_html=True,
    )
    report_missing_val = st.text_area(
        "Entrer ou modifier le text",
        "Les données extraites de la surveillance à Symphonia ne présentent pas de valeurs manquantes conventionnelles, mais plutôt des valeurs telles que null ou 0. En termes de comptage, cela indique qu'aucun objet n'a franchi une ligne de comptage ou n'a été détecté pendant certaines périodes. Ces valeurs sont essentielles pour comprendre l'activité réelle et les périodes d'inactivité dans les zones surveillées.",
    )
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">2. Préparation et Transformation des Données</h4>',
        unsafe_allow_html=True,
    )
    report_data_transform = st.text_area(
        "Entrer ou modifier le text",
        "Pour la préparation des données, les informations extraites par scénario ont été agrégées par caméra, puis centralisées pour un traitement global. La colonne Horodatage a subi des conversions pour faciliter l'analyse des tendances temporelles. Cette transformation des données permet une visualisation plus claire et une analyse approfondie des modèles de circulation et d'utilisation des espaces à Symphonia.",
    )

    # Descriptive Statistics
    st.subheader("IV. Statistiques Descriptives")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">1. Résumé Général des Données</h4>',
        unsafe_allow_html=True,
    )
    
    resume_gene_data = st.text_area("Entrer ou Modifier le résumé sur les données","Pour mieux appréhender les données extraites de la vidéosurveillance à Symphonia, une analyse statistique clé est essentielle. Cette section présente les principales mesures de tendance centrale et de dispersion pour les variables pertinentes, telles que le nombre d'objets détectés par caméra.")
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">Statistique Récaputilative </h5>',
        unsafe_allow_html=True,
    )
    
    report_desc_info = st.dataframe(summary_desc_symphonia(symphonia_data))

    # distribution des donnees.
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">2.Distribution des donnees</h4>',
        unsafe_allow_html=True,
    )

    counts = count_by_type_objet(symphonia_data)
    
    # Accessing the counts
    bus_count = counts['bus']
    moto_count = counts['moto']
    personne_count = counts['personne']
    vehicule_intermediaire_count = counts['véhicule intermédiaire']
    vehicule_leger_count = counts['véhicule léger']
    velo_count = counts['vélo']
    poids_lourd_count = counts['poids lourd']

    st.markdown("""
    <style>
    .centered-image {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    t_o_col1, t_o_col2, t_o_col3, t_o_col4 = st.columns(4)
    s_o_col1, s_o_col2, s_o_col3 = st.columns([1, 1, 1])
    
    bus_image_path = 'pages/icones/bus.png'
    moto_image_path = 'pages/icones/moto.png'
    personne_image_path = 'pages/icones/personne.png'
    vehicule_intermediaire_image_path ='pages/icones/vehicule_intermediare.png'
    vehicule_leger_image_path ='pages/icones/vehicule_leger.png'
    velo_image_path ='pages/icones/velo.png'
    poid_lourd_path ='pages/icones/poid_lourd.png'
    
    with st.container():
        
        with t_o_col1:
            st.image(bus_image_path)
            st.markdown(
            f'<h5 style="text-align: center; color: #000000;">Bus: {bus_count}</h5>',
            unsafe_allow_html=True,
        )
        with t_o_col2:
            st.image(moto_image_path)
            st.markdown(
            f'<h5 style="text-align: center; color: #000000;">Moto: {moto_count}</h5>',
            unsafe_allow_html=True,
        )
        with t_o_col3:
            st.image(personne_image_path)
            st.markdown(
            f'<h5 style="text-align: center; color: #000000;">Personne: {personne_count}</h5>',
            unsafe_allow_html=True,
        )
        
        with t_o_col4:
            st.image(velo_image_path)
            st.markdown(
            f'<h5 style="text-align: center; color: #000000;">Vélo: {velo_count}</h5>',
            unsafe_allow_html=True,
        )
            
        with s_o_col1:
            st.image(vehicule_leger_image_path)
            st.markdown(
            f'<h5 style="text-align: center; color: #000000;">Véhicule Léger: {vehicule_leger_count}</h5>',
            unsafe_allow_html=True,
        )
        with s_o_col2:  
            st.image(vehicule_intermediaire_image_path)
            st.markdown(
            f'<h5 style="text-align: center; color: #000000;">Véhicule Intermédiaire: {vehicule_intermediaire_count}</h5>',
            unsafe_allow_html=True,
        )
        with s_o_col3:
            st.image(poid_lourd_path, use_column_width=None)
            st.markdown(
            f'<h5 style="text-align: center; color: #000000;">Poids Lourd: {poids_lourd_count}</h5>',
            unsafe_allow_html=True,
        )

    if symphonia_data is not None:
        distribution_fig = px.histogram(
            symphonia_data,
            x="Type d'objet",
            y="Nombre",
            title="Distribution des données",
        )
        st.plotly_chart(distribution_fig)

    # Descriptive stat tendance Temporelle
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">3. Tendances et Model Temporelles</h4>',
        unsafe_allow_html=True,
    )
    
    # Trends and patterns
    mwd_fig, image_stream_mwd = plot_weekly_daily_patterns_real_estate(symphonia_data, "Horodatage", "Nombre")
    st.plotly_chart(mwd_fig)
    
    # Descriptive Stat, Variabilite et dispersion.
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">4. Variabilité et Dispersions</h4>',
        unsafe_allow_html=True,
    )
    
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">Variabilité entre les Caméras</h5>',
        unsafe_allow_html=True,
    )
    st.write("Analyse Comparative : Comparaison du nombre moyen d'objets détectés par caméra")
    ana_compa_cam_fig, image_stream_compa_fig = analyse_comparative_par_camera(symphonia_data)
    st.plotly_chart(ana_compa_cam_fig)
    
    st.write("Comparaison du nombre moyen d'objets détectés par caméra")
    comparaison_fig, image_stream_comparaison = plot_comparaison_type_objet_real_estate(symphonia_data)
    st.plotly_chart(comparaison_fig)
    
    st.write("Écart-type intercaméras")
    ecart_type_fig, image_stream_ecart_type = ecart_type_intercameras(symphonia_data)
    st.plotly_chart(ecart_type_fig)
    
    st.write("Variabilité entre les Caméras")
    rep_cam_fig, rep_scen_fig, fig_categorie,image_stream_cam,image_stream_cat, image_stream_scen = repartition_viz(symphonia_data)

    
    st.plotly_chart(rep_cam_fig)
    
    dia_disp_fig, image_stream_diagram_dispersion = diagrammes_de_dispersion(symphonia_data)
    st.plotly_chart(dia_disp_fig)
    
    st.write("Variabilité entre les Types d'Objets Détectés")
    analyse_corr_fig, image_stream_analyse_corr_fig = analyse_correlation(symphonia_data)
    st.plotly_chart(analyse_corr_fig)
    
    # Correlation 
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">5. Corrélations</h4>',
        unsafe_allow_html=True,
    )
    corr_analysis_fig = correlation_analysis(symphonia_data)
    st.plotly_chart(corr_analysis_fig)
    
    # Anomalie et Point Atypiques
    # Anomalies and Insight
    st.subheader("6. Detection d'anomalies")
    
    anomaly_plot, anomaly_summary, image_stream_anomaly_plot = analyze_anomalies(symphonia_data, threshold=0.01)
    st.plotly_chart(anomaly_plot)
    st.write(anomaly_summary)
    
    # Segmentation et Groupement
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">6. Segmentation et Groupement</h4>',
        unsafe_allow_html=True,
    )
    
    seg_group_fig, clusters, image_stream_seg_group_fig = segmentation_grouping(symphonia_data, n_clusters=5)
    st.plotly_chart(seg_group_fig)
    
    # Interpretation et synthese
    
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">7. Synthèse et Interprétation</h4>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">1. Résumé des Observations Clés :</h5>',
        unsafe_allow_html=True,
    )
    summary, interpretation = main_analysis(symphonia_data)
    def dict_to_string(d):
        return '\n'.join(f"{k}: {v}" for k, v in d.items())

    summary_string = dict_to_string(summary)
    interpretation_string = dict_to_string(interpretation)
    
    summary_text = st.text_area("Entrer ou modifier le resumer",
                                 f"Les données montrent une variabilité significative dans le nombre d'objets détectés par les différentes caméras. Certaines caméras, situées dans des zones à forte circulation, enregistrent des volumes de détection plus élevés.\n"
             f"\n"
    f"Les types d'objets détectés varient également selon les périodes de la journée, avec une augmentation notable des personnes détectées pendant les heures de pointe et une présence accrue de véhicules lourds pendant les heures creuses.\n"
    f"\n"
    f"Les analyses de segmentation révèlent des groupes distincts d'objets détectés en fonction des heures et des jours de la semaine, suggérant des schémas d\'activité réguliers.")
    
    st.write("Résumé des Observations Clés :", summary)
    
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">2. Interprétation des Résultats :</h5>',
        unsafe_allow_html=True,
    )
    
    interpretation_text =st.text_area("Entrer ou Modifier l'interpretation", f"La variabilité entre les caméras peut s'expliquer par leur emplacement stratégique dans les zones à haute circulation, comme les entrées principales et les parkings.\n"
             f"\n"
            f"Les différences dans les types d'objets détectés au cours de la journée indiquent des habitudes de mouvement spécifiques, comme les trajets domicile-travail pour les personnes et les opérations de livraison pour les véhicules lourds.\n"
            f"\n"
            f"Les segments identifiés à partir des caractéristiques temporelles suggèrent des moments spécifiques où la sécurité doit être renforcée ou où une attention particulière est nécessaire.\n")
    
    st.write("Interprétation des Résultats :", interpretation)
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">3. Lien avec les Objectifs du Rapport :</h5>',
        unsafe_allow_html=True,
    )
    lien_rapport=st. text_area("Entrer ou Modifier la sexion", f"Les observations montrent que les caméras sont efficaces pour surveiller les zones à forte circulation et que les types d'objets détectés varient selon les périodes, répondant ainsi aux objectifs de surveillance générale de Symphonia. \n"
                  f"L\'identification des segments temporels et des zones à forte activité permet de formuler des recommandations pour améliorer la sécurité et l\'efficacité des opérations de surveillance.")
    
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">4. Conclusions Préliminaires :</h5>',
        unsafe_allow_html=True,
    )
    
    conclusion_pre=st.text_area(" Entrer ou Modifier la conclusion preliminaire", f"Les données de vidéosurveillance de Symphonia révèlent des schémas d'activité distincts selon les zones et les périodes. Les caméras stratégiquement placées jouent un rôle crucial dans la détection des objets. \n Des mesures spécifiques peuvent être mises en place pour renforcer la sécurité pendant les périodes de haute activité et pour optimiser l'utilisation des ressources de surveillance. \n  Ces conclusions préliminaires servent de base pour des analyses futures plus détaillées et pour l'élaboration de recommandations concrètes visant à améliorer la sécurité et la gestion des ressources à Symphonia.") 
    
    


    # Conclusion
    st.subheader("V. Conclusion Générale")
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">5.1. Résumé</h5>',
        unsafe_allow_html=True,
    )
    conclusion_text = st.text_area(f"Entrer ou Modifier le text", "L'analyse des données de vidéosurveillance de la propriété immobilière Symphonia, située à Abidjan Cocody, a révélé des informations cruciales sur les schémas de détection des différents types d'objets. Grâce à l'application Videtics, nous avons pu recueillir des données précises sur la fréquence et la répartition des objets franchissant les lignes de comptage, détectés par catégorie et type d'objet. \n"
                                   f"les principales observations sont les suivantes: \n"
                                   f"Variabilité Temporelle : Les tendances mensuelles, hebdomadaires et journalières ont montré des variations significatives dans la détection des objets, indiquant des périodes de haute et de faible activité.\n"
                                   f"Distribution par Type d'Objet : Les véhicules légers et les personnes sont les types d'objets les plus fréquemment détectés, avec des variations notables entre les différentes périodes. \n"
                                   f"Variabilité entre les Caméras : Certaines caméras ont capté une activité nettement plus élevée que d'autres, soulignant l'importance de leur emplacement stratégique. \n"
                                   f"Anomalies et Points Atypiques : L'analyse a permis d'identifier des anomalies dans les schémas de détection, indiquant des événements inhabituels ou des erreurs potentielles dans les données.")
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">5.2. Recommendation</h5>',
        unsafe_allow_html=True,
    )
    recommendation_text = st.text_area(f"Modifier le text", "Sur la base des analyses effectuées et des observations obtenues, les recommandations suivantes sont proposées : \n"
                                       f"Optimisation du Placement des Caméras : Revoir les emplacements des caméras pour maximiser la couverture des zones à haute activité et réduire les angles morts. Les caméras dans les zones de faible activité pourraient être repositionnées. \n"
                                       f"Surveillance Renforcée aux Périodes de Haute Activité : Augmenter la vigilance et la surveillance pendant les périodes identifiées de haute activité pour prévenir les incidents et améliorer la sécurité. \n"
                                       f"Analyse Continue des Anomalies : Mettre en place un système de surveillance continue des anomalies pour identifier rapidement les événements inhabituels et prendre des mesures correctives.\n"
                                       f"Amélioration des Capacités de Détection : Envisager des mises à jour ou des améliorations technologiques pour les caméras et les algorithmes de détection afin de renforcer la précision et la fiabilité des données collectées.")
    point_final = st.text_area("Modifier le point final de la conclusion","En conclusion, cette analyse offre une vue d'ensemble détaillée des schémas de détection des objets sur le site de Symphonia, fournissant des informations précieuses pour améliorer la sécurité et l'efficacité de la surveillance. Les recommandations proposées visent à optimiser l'utilisation des ressources disponibles et à renforcer les mesures de sécurité en place.")

    if st.button("Générer rapport"):
        pdf_data = generate_report(
            title=report_title,
            subtitle=report_subtitle,
            intro_title="I. INTRODUCTION",
            intro_brief = report_brief_overview,
            title_objectif="1.Objectif",
            intro_objectif=report_objectif,
            title_scope="2. Scope",
            text_scope=report_scope,
            title_data_desc="II. Description des donnees",
            sub_title_data_text="1. Aperçu de l'ensemble des donnees",
            data_overview_text=report_data_overview,
            sub_title_data_desc_col="2. Description des colonnes des données",
            ms_subtitle_def_typob ="3. Définition des types d\'objets",
            text_def_to = intro_def_TO,
            data_cleaning_title="III. Nettoyage et Preparation des donnee",
            missing_val_title="1. Gestion des Valeurs Manquantes",
            missing_val_content=report_missing_val,
            transformation_data_title="2. Préparation et Transformation des Données.",
            transformation_data_content=report_data_transform,
            desc_stat_title="IV. Statistiques Descriptives",
            desc_stat_subtite ="1. Résumé Général des Données",
            ms_distribution_title ="2.Distribution des données",
            ms_tendances_temporelles ="3.Tendances et Model Temporelles",
            desc_stat_rgd_content= resume_gene_data,
            desc_stat_content=report_desc_info,
            data_viz = "4. Variabilité et Dispersions",
            ms_variabilite_cameras= "a. Variabilité entre les Caméras",
            ms_ecart_type_intercam = "b. Écart-type intercaméras",
            ms_repartition_camera = "c. Repartition des données par Caméras",
            ms_corr_analysis = "5. Analyse de correlation",
            ms_anomali_detection ="6. Detection d'anomalies",
            ms_seg_group="7. Segmentation et Groupement",
            
            
            # interpretation et synthese
            # summary
            ms_interpretation_synthese_title = "8. Synthèse et Interprétation",
            ms_summary_subtitle ="1. Résumé des Observations Clés :",
            ms_summary_text = summary_text,
            ms_summary_result = f"Résumé des Observations Clés :\n{summary_string}",
            
            #interpretation
            ms_interpretation_subtitle="2. Interprétation des Résultats :",
            ms_interpretation_text = interpretation_text,
            ms_interpretation_result = f"Interprétation des Résultats :\n{interpretation_string}",
            
            # lien avec le rapport
            ms_lien_rapport_subtitle="3. Lien avec les Objectifs du Rapport :",
            ms_lien_rapport =lien_rapport,
            
            #conclusion preliminaire
            ms_conclusison_pre_subtitle="4. Conclusions Préliminaires :",
            ms_conclusion_pre = conclusion_pre,
            
            #conclusion
            conclusion_title="V. Conclusion Générale",
            conclusion_subtitl_r="5.1. Résumé",
            resume_content=conclusion_text,
            recomm_title="5.2. Recommendation",
            recomm_content=recommendation_text,
            ms_point_final = point_final,
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
    st.subheader("I. Introduction")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">1.1. Objectif</h4>',
        unsafe_allow_html=True,
    )
    report_objectif = st.text_area(
        "Entrez ou modifier l'objectif",
        "L'objectif de ce rapport est de fournir une analyse détaillée des données capturées par les caméras de surveillance installées à Symphonia, une propriété immobilière située à Abidjan, Cocody. Cette analyse vise à offrir des insights sur la performance, les tendances et les anomalies détectées par chaque caméra.",
    )

    st.markdown(
        '<h4 style="text-align: left; color: #000000;">1.2. Scope</h4>',
        unsafe_allow_html=True,
    )
    report_scope = st.text_area(
        "Entrez ou modifier le Scope",
        "Ce rapport se concentre sur les données collectées par les caméras individuelles, en examinant divers aspects tels que les comptes de détection, la distribution des types d'objets, les schémas spatiaux et temporels, les anomalies et la fiabilité des caméras. Les insights tirés de cette analyse aideront à optimiser le système de surveillance et à renforcer les mesures de sécurité.",
    )

    # Data description
    st.subheader("II. Description des donnees")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">2.1. Aperçu des Données</h4>',
        unsafe_allow_html=True,
    )
    report_data_overview = st.text_area(
        "Entrez ou modifier le text",
        "Les données comprennent l'analyse des séquences de surveillance collectées à l'aide de l'application Videtics, qui utilise la technologie de vision par ordinateur pour la détection d'objets, le comptage et la détection d'intrusions.",
    )
    
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">2.2. Description des colonnes des données</h4>',
        unsafe_allow_html=True,
    )
    st.markdown("""
    - **Horodatage** : Cette colonne indique la date et l'heure précises de chaque événement enregistré. Elle est essentielle pour analyser les tendances temporelles et les schémas de comportement.
    - **Caméra** : Identifie la caméra spécifique ayant capturé l'événement. Cela permet de localiser géographiquement les événements et de comprendre les zones de la propriété où l'activité est la plus dense.
    - **Scénario** : Décrit le contexte ou la situation dans laquelle l'événement s'est produit, aidant ainsi à catégoriser les différentes situations de surveillance.
    - **Catégorie** : Classe les événements en différentes catégories, facilitant ainsi une analyse segmentée des données.
    - **Type d'objet** : Indique le type d'objet détecté, par exemple, personne, moto, véhicule léger, etc. Cette colonne est cruciale pour comprendre la distribution des différents objets sur la propriété.
    - **Nombre** : Quantifie le nombre d'objets détectés pour chaque enregistrement, permettant une analyse quantitative des données.
    """)
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">2.3. Définition des types d\'objets</h4>',
        unsafe_allow_html=True,
    )
    st.write("Les types d'objets détectés et enregistrés par le système Videtics sont variés, couvrant les principaux éléments susceptibles de se déplacer dans la propriété Symphonia :")
    st.markdown("""
    - **Bus** : Véhiculent des passagers en grands nombres et nécessitent une surveillance particulière en raison de leur taille et de leur impact potentiel sur la circulation.
    - **Moto** : Inclut toutes les formes de motocyclettes, souvent plus rapides et plus difficiles à détecter que les véhicules plus grands.
    - **Personne** : Comprend tous les piétons, offrant des insights sur les mouvements des résidents, visiteurs et personnels.
    - **Véhicule intermédiaire** : Réfère aux véhicules de taille moyenne, tels que les camionnettes, qui jouent un rôle clé dans la logistique et le transport.
    - **Véhicule léger** : Inclut les voitures personnelles et autres véhicules de petite taille, couramment utilisés par les résidents et visiteurs.
    - **Vélo** : Inclut toutes les formes de bicyclettes, de plus en plus courantes dans les zones résidentielles et nécessitant une surveillance pour la sécurité.
    - **Poid lourd** : Véhicule de grande taille destiné au transport de marchandises, caractérisé par une capacité de charge élevée.
    """)

    # Data cleaning and Preparation
    st.subheader("III. Nettoyage et Préparation des Données")
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">3.1. Valeurs Manquantes</h4>',
        unsafe_allow_html=True,
    )
    report_missing_val = st.text_area(
        "Entrez ou modifier le text",
        "Les données extraites ne contiennent pas de valeurs manquantes mais peuvent inclure des valeurs nulles ou zéro, indiquant qu'aucun événement de détection d'objet n'a eu lieu.",
    )
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">3.2 Préparation et Transformation des Données</h4>',
        unsafe_allow_html=True,
    )
    report_data_transform = st.text_area(
        "Entrez ou modifier le text",
        "Les horodatages ont été convertis au format datetime et les colonnes non pertinentes ont été supprimées.",
    )

    # Descriptive Statistics
    st.subheader("4. Statistiques Descriptives")
    st.subheader("4.1. Résumé Général des Données")
    st.text_area("Entrer ou Modifier le resume","Statistiques clés telles que la moyenne, la médiane, l'écart-type, le minimum et le maximum pour le nombre d'objets détectés par caméra.")
    report_desc_info = st.dataframe(summary_desc_symphonia(selected_df))

    st.markdown(
        '<h4 style="text-align: left; color: #000000;">4.2. Variabilité et Dispersions</h4>',
        unsafe_allow_html=True,
    )
    
    fig_mean, fig_std = analyze_scenario_variability(selected_df)
    st.plotly_chart(fig_mean)
    st.plotly_chart(fig_std)
    
    st.subheader("5.1.  Distribution des Types d'Objets")
    
    counts = count_by_type_objet(selected_df)
    
    # Accessing the counts
    bus_count = counts['bus']
    moto_count = counts['moto']
    personne_count = counts['personne']
    vehicule_intermediaire_count = counts['véhicule intermédiaire']
    vehicule_leger_count = counts['véhicule léger']
    velo_count = counts['vélo']
    poids_lourd_count = counts['poids lourd']

    st.markdown("""
    <style>
    .centered-image {
        display: flex;
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    t_o_col1, t_o_col2, t_o_col3, t_o_col4 = st.columns(4)
    s_o_col1, s_o_col2, s_o_col3 = st.columns(3, gap="small")
    
    bus_image_path = 'pages/icones/bus.png'
    moto_image_path = 'pages/icones/moto.png'
    personne_image_path = 'pages/icones/personne.png'
    vehicule_intermediaire_image_path ='pages/icones/vehicule_intermediare.png'
    vehicule_leger_image_path ='pages/icones/vehicule_leger.png'
    velo_image_path ='pages/icones/velo.png'
    poid_lourd_path ='pages/icones/poid_lourd.png'
    
    with t_o_col1:
        st.image(bus_image_path)
        st.markdown(
        f'<h5 style="text-align: center; color: #000000;">Bus: {bus_count}</h5>',
        unsafe_allow_html=True,
    )
    with t_o_col2:
        st.image(moto_image_path)
        st.markdown(
        f'<h5 style="text-align: center; color: #000000;">Moto: {moto_count}</h5>',
        unsafe_allow_html=True,
    )
    with t_o_col3:
        st.image(personne_image_path)
        st.markdown(
        f'<h5 style="text-align: center; color: #000000;">Personne: {personne_count}</h5>',
        unsafe_allow_html=True,
    )
    
    with t_o_col4:
        st.image(velo_image_path)
        st.markdown(
        f'<h5 style="text-align: center; color: #000000;">Vélo: {velo_count}</h5>',
        unsafe_allow_html=True,
    )
        
    with s_o_col1:
        st.image(vehicule_leger_image_path)
        st.markdown(
        f'<h5 style="text-align: center; color: #000000;">Véhicule Léger: {vehicule_leger_count}</h5>',
        unsafe_allow_html=True,
    )
    with s_o_col2:  
        st.image(vehicule_intermediaire_image_path)
        st.markdown(
        f'<h5 style="text-align: center; color: #000000;">Véhicule Intermédiaire: {vehicule_intermediaire_count}</h5>',
        unsafe_allow_html=True,
    )
    with s_o_col3:
        st.image(poid_lourd_path, use_column_width=None)
        st.markdown(
        f'<h5 style="text-align: center; color: #000000;">Poids Lourd: {poids_lourd_count}</h5>',
        unsafe_allow_html=True,
    )


    if selected_df is not None:
        distribution_fig = px.histogram(
            selected_df,
            x="Type d'objet",
            y="Nombre",
            title="Distribution des données",
        )
        st.plotly_chart(distribution_fig)
    
    
    #repartition des type d'objet par mois
    
    st.subheader("5.2. Repartition et Comaparaison des données")
    st.subheader("5.2.1. Repartition des données par mois")
    # Preprocess the data
    symphonia_data = preprocess_data(selected_df)
    
    # Create the pivot table
    pivot_table = create_pivot_table(selected_df)
    
    # Display the table in Streamlit
    st.write(pivot_table)
    
    # comparaison des donnees 
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">5.2.2. comparaison des données</h4>',
        unsafe_allow_html=True,
    )
    comparaison_fig, image_stream_comparaison = plot_comparaison_type_objet_real_estate(selected_df)
    st.plotly_chart(comparaison_fig)

    # Trends and patterns
    st.subheader("5.3. Schémas Temporels")
    mwd_fig, image_stream_mwd = plot_weekly_daily_patterns_real_estate(selected_df, "Horodatage", "Nombre")
    st.plotly_chart(mwd_fig)

    # Anomalies and Insight
    st.subheader("5.4.  Anomalies et Événements Inhabituels")
    anomaly_plot, anomaly_summary, image_stream_anomaly_plot = analyze_anomalies(selected_df)
    st.plotly_chart(anomaly_plot)
    
    # synthese et interpretation
    st.markdown(
        '<h4 style="text-align: left; color: #000000;">6. Synthèse et Interprétation</h4>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">6.1. Résumé des Observations Clés :</h5>',
        unsafe_allow_html=True,
    )
    summary, interpretation = main_analysis_per_cam(symphonia_data)
    def dict_to_string(d):
        return '\n'.join(f"{k}: {v}" for k, v in d.items())

    summary_string = dict_to_string(summary)
    interpretation_string = dict_to_string(interpretation)
    
    summary_text = st.text_area("Entrer ou modifier le resumer",
                                 f"Les données montrent une variabilité significative dans le nombre d'objets détectés par les différentes caméras. Certaines caméras, situées dans des zones à forte circulation, enregistrent des volumes de détection plus élevés.\n"
             f"\n"
    f"Les types d'objets détectés varient également selon les périodes de la journée, avec une augmentation notable des personnes détectées pendant les heures de pointe et une présence accrue de véhicules lourds pendant les heures creuses.\n"
    f"\n"
    f"Les analyses de segmentation révèlent des groupes distincts d'objets détectés en fonction des heures et des jours de la semaine, suggérant des schémas d\'activité réguliers.")
    
    st.write("Résumé des Observations Clés :", summary)
    
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">6.2. Interprétation des Résultats :</h5>',
        unsafe_allow_html=True,
    )
    
    interpretation_text =st.text_area("Entrer ou Modifier l'interpretation", f"La variabilité entre les caméras peut s'expliquer par leur emplacement stratégique dans les zones à haute circulation, comme les entrées principales et les parkings.\n"
             f"\n"
            f"Les différences dans les types d'objets détectés au cours de la journée indiquent des habitudes de mouvement spécifiques, comme les trajets domicile-travail pour les personnes et les opérations de livraison pour les véhicules lourds.\n"
            f"\n"
            f"Les segments identifiés à partir des caractéristiques temporelles suggèrent des moments spécifiques où la sécurité doit être renforcée ou où une attention particulière est nécessaire.\n")
    
    st.write("Interprétation des Résultats :", interpretation)

    # Conclusion
    st.subheader("7. Conclusion Générale")
    st.write("7.1. Résumé")
    conclusion_text = st.text_area("Entrer ou Modifier le text", "")
    st.markdown(
        '<h5 style="text-align: left; color: #000000;">7.2. Recommendation</h5>',
        unsafe_allow_html=True,
    )
    recommendation_text = st.text_area("Modifier le text", "")

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
    st.sidebar.subheader("Type de rapport")
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
        
    # Streamlit sidebar configuration
    st.sidebar.subheader("Fichier Généré")
    selected_file = st.sidebar.selectbox("Sélectionnez un fichier", files)

    # Display the download button for the selected file
    if selected_file:
        file_path = os.path.join(folder_path2, selected_file)
        file_extension = selected_file.split('.')[-1]
        
        with open(file_path, "rb") as file:
            file_data = file.read()
        
        if file_extension == 'pdf':
            st.sidebar.download_button(
                label="Télécharger le fichier PDF",
                data=file_data,
                file_name=selected_file,
                mime="application/pdf"
            )
        elif file_extension == 'docx':
            st.sidebar.download_button(
                label="Télécharger le fichier DOCX",
                data=file_data,
                file_name=selected_file,
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )


if __name__ == "__main__":
    main()
