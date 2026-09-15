import csv
import sys
from pathlib import Path


MEMORY_BANDWIDTH_GB_S = 484.0
FP32_PEAK_GFLOP_S = 10600.0


def parse_float(value):
    value = value.strip()
    if not value:
        return None
    return float(value)


def classify(ai, measured_gflops):
    memory_roof = ai * MEMORY_BANDWIDTH_GB_S
    attainable = min(memory_roof, FP32_PEAK_GFLOP_S)

    if attainable <= 0.0:
        return memory_roof, attainable, "invalid"

    efficiency = measured_gflops / attainable

    if memory_roof < FP32_PEAK_GFLOP_S:
        bound = "memory roof"
    else:
        bound = "compute roof"

    if efficiency < 0.35:
        return memory_roof, attainable, f"below {bound}; investigate overhead"

    return memory_roof, attainable, f"near {bound}"


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 roofline_from_csv.py roofline_template.csv")
        return 1

    csv_path = Path(sys.argv[1])
    rows = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            kernel = row["kernel"].strip()
            ai = parse_float(row["arithmetic_intensity_flop_per_byte"])
            measured = parse_float(row["measured_gflops"])
            notes = row.get("notes", "").strip()

            if ai is None or measured is None:
                rows.append({
                    "kernel": kernel,
                    "ai": None,
                    "measured": None,
                    "memory_roof": None,
                    "attainable": None,
                    "classification": "TODO: fill missing values",
                    "notes": notes,
                })
                continue

            memory_roof, attainable, classification = classify(ai, measured)

            rows.append({
                "kernel": kernel,
                "ai": ai,
                "measured": measured,
                "memory_roof": memory_roof,
                "attainable": attainable,
                "classification": classification,
                "notes": notes,
            })

    report_path = Path("roofline_report.md")

    with report_path.open("w", newline="\n") as handle:
        handle.write("# Roofline Report\n\n")
        handle.write(f"Memory bandwidth ceiling: {MEMORY_BANDWIDTH_GB_S:.1f} GB/s\n\n")
        handle.write(f"FP32 compute ceiling: {FP32_PEAK_GFLOP_S:.1f} GFLOP/s\n\n")
        handle.write("| Kernel | AI (FLOP/byte) | Measured GFLOP/s | "
                     "Memory roof | Attainable roof | Classification | Notes |\n")
        handle.write("| ------ | --------------: | ---------------: | "
                     "----------: | --------------: | -------------- | ----- |\n")

        for row in rows:
            if row["ai"] is None:
                handle.write(
                    f"| {row['kernel']} |  |  |  |  | "
                    f"{row['classification']} | {row['notes']} |\n"
                )
                continue

            handle.write(
                f"| {row['kernel']} | {row['ai']:.3f} | "
                f"{row['measured']:.3f} | {row['memory_roof']:.3f} | "
                f"{row['attainable']:.3f} | {row['classification']} | "
                f"{row['notes']} |\n"
            )

    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
