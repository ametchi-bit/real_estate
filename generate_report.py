from fpdf import FPDF
from io import BytesIO
import pandas as pd
from datetime import datetime


class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, self.title, 0, 1, "C")

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")

    def add_section(self, title, text):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, title, 0, 1)
        self.set_font("Arial", "", 12)
        if isinstance(text, str):
            self.multi_cell(0, 10, text)
        else:
            self.multi_cell(0, 10, str(text))
        self.ln(10)


def generate_report(
    title,
    subtitle,
    objective,
    scope,
    overview,
    columns,
    missing_values,
    transformations,
    summary_stats,
    data_distribution,
    traffic_volume_cam,
    traffic_volume_scen,
    scenarios,
    monthly_weekly_daily_patterns,
    anomaly_detection,
    key_insights,
    summary,
    recommendations,
    appendices,
):
    pdf = PDF()
    pdf.add_page()
    pdf.set_title(title)

    # Adding sections to the PDF
    pdf.add_section("Title", title)
    pdf.add_section("Subtitle", subtitle)
    pdf.add_section("Objective", objective)
    pdf.add_section("Scope", scope)
    pdf.add_section("Overview", overview)
    pdf.add_section("Columns", columns)
    pdf.add_section("Missing Values", missing_values)
    pdf.add_section("Transformations", transformations)
    pdf.add_section("Summary Stats", summary_stats)
    pdf.add_section("Data Distribution", "See attached images")
    pdf.add_section("Traffic Volume by Camera", "See attached images")
    pdf.add_section("Traffic Volume by Scenario", "See attached images")
    pdf.add_section("Scenarios", "See attached images")
    pdf.add_section("Monthly, Weekly, Daily Patterns", "See attached images")
    pdf.add_section("Anomaly Detection", anomaly_detection)
    pdf.add_section("Key Insights", key_insights)
    pdf.add_section("Summary", summary)
    pdf.add_section("Recommendations", recommendations)
    pdf.add_section("Appendices", appendices)

    # Add images
    pdf.image(data_distribution, x=10, y=None, w=180)
    pdf.add_page()
    pdf.image(traffic_volume_cam, x=10, y=None, w=180)
    pdf.add_page()
    pdf.image(traffic_volume_scen, x=10, y=None, w=180)
    pdf.add_page()
    pdf.image(scenarios, x=10, y=None, w=180)
    pdf.add_page()
    pdf.image(monthly_weekly_daily_patterns, x=10, y=None, w=180)

    # Convert PDF to bytes
    pdf_output = BytesIO()
    pdf.output(pdf_output)
    pdf_output.seek(0)
    return pdf_output

    # output = BytesIO()
    # pdf.output(desr="s").encode("latin1")
    # output.write(pdf.output(dest="S").encode("latin1"))
    # output.seek(0)
    # return output.getvalue()
