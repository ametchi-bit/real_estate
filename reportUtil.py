from fpdf import FPDF
from io import BytesIO


class PDFReport(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Surveillance Data Analysis Report", 0, 1, "C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def write_section_title(self, title):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, title, 0, 1, "L")
        self.ln(10)

    def write_paragraph(self, text):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, text)
        self.ln(5)


def generate_report(**kwargs):
    pdf = PDFReport()

    write_introduction(pdf, kwargs["objective"], kwargs["scope"])
    write_data_description(pdf, kwargs["overview"], kwargs["columns"])
    write_data_cleaning(pdf, kwargs["missing_values"], kwargs["transformations"])
    write_descriptive_statistics(
        pdf, kwargs["summary_stats"], kwargs["data_distribution"]
    )
    write_visualization(pdf, kwargs["traffic_volume"], kwargs["scenarios"])
    write_trends_patterns(pdf, kwargs["monthly_trend"], kwargs["weekly_daily_patterns"])
    write_anomalies_insights(pdf, kwargs["anomaly_detection"], kwargs["key_insights"])
    write_conclusion(pdf, kwargs["summary"], kwargs["recommendations"])
    write_appendices(pdf, kwargs["appendices"])

    output = BytesIO()
    pdf.output(output)
    pdf_data = output.getvalue()
    output.close()

    return pdf_data


def write_introduction(pdf, objective, scope):
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Introduction", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Objectif: {objective}")
    pdf.multi_cell(0, 10, f"Scope: {scope}")


def write_data_description(pdf, overview, columns):
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Description des données", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Aperçu: {overview}")
    pdf.multi_cell(0, 10, f"Colonnes: {columns}")


def write_data_cleaning(pdf, missing_values, transformations):
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Nettoyage des données", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Valeurs Manquantes: {missing_values}")
    pdf.multi_cell(0, 10, f"Transformations: {transformations}")


def write_descriptive_statistics(pdf, summary_stats, data_distribution):
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Statistiques Descriptives", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Statistiques Résumé: {summary_stats}")
    pdf.multi_cell(0, 10, f"Distribution des Données: {data_distribution}")


def write_visualization(pdf, traffic_volume, scenarios):
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Visualisation des données", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Volume de Trafic: {traffic_volume}")
    pdf.multi_cell(0, 10, f"Scénarios: {scenarios}")


def write_trends_patterns(pdf, monthly_trend, weekly_daily_patterns):
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Tendances et Modèles", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Tendance Mensuelle: {monthly_trend}")
    pdf.multi_cell(
        0, 10, f"Modèles Hebdomadaires et Quotidiens: {weekly_daily_patterns}"
    )


def write_anomalies_insights(pdf, anomaly_detection, key_insights):
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Détection d'anomalies et Aperçus", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Détection d'anomalies: {anomaly_detection}")
    pdf.multi_cell(0, 10, f"Aperçus Clés: {key_insights}")


def write_conclusion(pdf, summary, recommendations):
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Conclusion", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Sommaire: {summary}")
    pdf.multi_cell(0, 10, f"Recommandations: {recommendations}")


def write_appendices(pdf, appendices):
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Appendices", 0, 1, "L")
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 10, f"Appendices: {appendices}")
