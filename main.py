import os
import sys
import traceback

from tools import to_acres, rule_shortlist, save_to_txt, UNIT_TO_ACRE
from vision import analyze_land, merge_analyses
from recommender import recommend
from tractor_web import get_tractor_examples
from pdf_report import build_pdf


def main():
    paths = input("Photo path(s), comma se alag: ").strip().strip('"').split(",")
    paths = [p.strip().strip('"') for p in paths if p.strip()]
    missing = [p for p in paths if not os.path.isfile(p)]
    if not paths or missing:
        print(f"Image file nahi mili: {missing or 'koi path nahi diya'}")
        return

    soil = input("Soil (black/clay/loam/sandy): ").strip().lower()
    crop = input("Crop: ").strip()
    unit = input(f"Unit ({'/'.join(UNIT_TO_ACRE)}): ").strip().lower() or "acre"
    try:
        acres = to_acres(float(input("Plot size: ").strip()), unit)
    except ValueError as e:
        print(f"Invalid size/unit: {e}")
        return
    if not soil or not crop:
        print("Soil aur crop dono zaruri hain.")
        return

    try:
        analysis = merge_analyses([analyze_land(p) for p in paths[:4]])
        if not analysis.image_quality_ok or analysis.confidence == "low":
            print("Photo clear nahi hai ya farmland nahi lag raha. Dobara clear photo do.")
            print(analysis.model_dump_json(indent=2))
            return

        shortlist = rule_shortlist(analysis, soil, acres, crop)
        result = recommend(analysis, shortlist, soil, acres, crop)
        tractors = get_tractor_examples(result.tractor_hp_range)
    except Exception:
        traceback.print_exc()
        return

    print(analysis.model_dump_json(indent=2))
    print(result.model_dump_json(indent=2))
    print("Tractors:", list(tractors) or tractors.note)

    with open("land_recommendation.pdf", "wb") as f:
        build_pdf(f, {"Soil": soil, "Crop": crop, "Plot size (acres)": acres},
                  analysis, result, tractors)
    print("PDF saved: land_recommendation.pdf")
    print(save_to_txt(result.model_dump_json(indent=2)))


if __name__ == "__main__":
    main()