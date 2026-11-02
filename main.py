"""CLI entry point. UI ke liye: streamlit run app.py"""
import traceback

from tools import to_acres, rule_shortlist
from vision import analyze_land
from recommender import recommend
from pdf_report import build_pdf
from tractor_web import get_tractor_examples


def main():
    image_path = input("Image path: ").strip().strip('"')
    soil = input("Soil type (clay/loam/sandy/black): ").strip()
    size = float(input("Plot size: "))
    unit = input("Unit (acre/bigha/hectare/sqft/sqm): ").strip() or "acre"

    try:
        acres = to_acres(size, unit)
    except ValueError as e:
        print(e)
        return

    crop = input("Intended crop: ").strip()

    try:
        analysis = analyze_land(image_path)
    except Exception:
        traceback.print_exc()
        return

    if not analysis.image_quality_ok or analysis.confidence.lower() == "low":
        print("Image clear nahi hai. Dusri photo upload karo (daylight, poora plot dikhe).")
        return

    shortlist = rule_shortlist(analysis, soil, acres, crop)
    result = recommend(analysis, shortlist, soil, acres, crop)

    tractors = get_tractor_examples(result.tractor_hp_range)
    pdf_path = "land_report.pdf"
    build_pdf(pdf_path,
              {"Soil": soil, "Crop": crop, "Plot size (acres)": acres},
              analysis, result, tractors)
    print(f"PDF saved: {pdf_path}")


if __name__ == "__main__":
    main()