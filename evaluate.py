"""Vision accuracy check.
Use:  python evaluate.py
test_cases/ me images rakho aur test_cases/expected.csv me sahi labels likho.
Output: terminal table + evaluation_results.md
"""
import csv
import os

from vision import analyze_land

FOLDER = "test_cases"
FIELDS = ["terrain", "vegetation_density", "surface_wetness"]


def main():
    rows, correct, total = [], {f: 0 for f in FIELDS}, {f: 0 for f in FIELDS}
    reject_ok = reject_total = 0

    with open(os.path.join(FOLDER, "expected.csv"), encoding="utf-8") as f:
        cases = list(csv.DictReader(f))

    for c in cases:
        path = os.path.join(FOLDER, c["filename"])
        if not os.path.exists(path):
            print(f"[skip] {c['filename']} nahi mili")
            continue
        try:
            a = analyze_land(path)
        except Exception as e:
            rows.append((c["filename"], f"ERROR: {e}", "-", "-"))
            continue

        rejected = (not a.image_quality_ok) or a.confidence.lower() == "low"

        if c["expect_reject"].strip().lower() == "yes":
            reject_total += 1
            ok = rejected
            reject_ok += ok
            rows.append((c["filename"], "reject expected",
                         "rejected" if rejected else f"accepted ({a.terrain}/{a.vegetation_density}/{a.surface_wetness})",
                         "OK" if ok else "WRONG"))
            continue

        marks = []
        for fld in FIELDS:
            exp, got = c[fld].strip().lower(), getattr(a, fld).strip().lower()
            total[fld] += 1
            ok = exp == got
            correct[fld] += ok
            marks.append(f"{fld}: exp={exp}, got={got} {'OK' if ok else 'WRONG'}")
        rows.append((c["filename"], "", " | ".join(marks), "rejected!" if rejected else ""))

    lines = ["# Vision evaluation", "", "| Image | Note | Result | Flag |", "|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} |" for r in rows]
    lines += ["", "## Accuracy"]
    for fld in FIELDS:
        if total[fld]:
            lines.append(f"- {fld}: {correct[fld]}/{total[fld]} ({100 * correct[fld] / total[fld]:.0f}%)")
    if reject_total:
        lines.append(f"- bad-image rejection: {reject_ok}/{reject_total}")

    out = "\n".join(lines)
    print(out)
    with open("evaluation_results.md", "w", encoding="utf-8") as f:
        f.write(out)
    print("\nSaved: evaluation_results.md")


if __name__ == "__main__":
    main()