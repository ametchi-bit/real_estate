import pandas as pd
from fpdf import FPDF
from io import BytesIO
from datetime import datetime


def export_to_xlsx(combined_df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine="xlsxwriter")
    combined_df.to_excel(writer, index=False, sheet_name="Sheet1")
    writer.save()
    output.seek(0)
    return output.getvalue()


def export_to_pdf(title, subtitle):
    pdf = FPDF()
    pdf.add_page()

    # Title Page
    pdf.set_font("Arial", size=24)
    pdf.cell(200, 20, txt=title, ln=True, align="C")

    pdf.set_font("Arial", size=18)
    pdf.cell(200, 20, txt=subtitle, ln=True, align="C")

    pdf.set_font("Arial", size=14)
    pdf.cell(
        200, 10, txt=f"Date: {datetime.now().strftime('%B %d, %Y')}", ln=True, align="C"
    )

    # Add additional report content here...

    output = BytesIO()
    pdf.output(dest="S").encode("latin1")  # Save to a string in Latin-1 encoding
    output.write(pdf.output(dest="S").encode("latin1"))
    output.seek(0)
    return output.getvalue()


def generate_report(title, subtitle, combined_df):
    pdf_data = export_to_pdf(title, subtitle)
    return pdf_data
