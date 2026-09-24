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


def classify_roof(ai, measured_gflops):
    if ai is None or measured_gflops is None:
        return "", "", "TODO: fill measured values"

    memory_roof = ai * MEMORY_BANDWIDTH_GB_S
    attainable = min(memory_roof, FP32_PEAK_GFLOP_S)

    if attainable <= 0.0:
        return memory_roof, attainable, "invalid"

    utilization = measured_gflops / attainable

    if memory_roof < FP32_PEAK_GFLOP_S:
        expected_limit = "memory"
    else:
        expected_limit = "compute"

    if utilization < 0.35:
        return memory_roof, attainable, f"below {expected_limit} roof"

    if utilization < 0.75:
        return memory_roof, attainable, f"moderate {expected_limit}-roof use"

    return memory_roof, attainable, f"near {expected_limit} roof"


def format_number(value):
    if value is None or value == "":
        return ""
    return f"{value:.3f}"


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 bottleneck_from_csv.py bottleneck_template.csv")
        return 1

    csv_path = Path(sys.argv[1])
    rows = []

    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)

        for row in reader:
            kernel = row["kernel"].strip()
            source = row["source_lab"].strip()
            ai = parse_float(row["arithmetic_intensity_flop_per_byte"])
            measured = parse_float(row["measured_gflops"])
            bandwidth = parse_float(row["measured_bandwidth_gbs"])
            symptom = row["observed_symptom"].strip()
            hypothesis = row["first_hypothesis"].strip()
            next_experiment = row["next_experiment"].strip()
            status = row["status"].strip()

            memory_roof, attainable, classification = classify_roof(
                ai, measured)

            rows.append({
                "kernel": kernel,
                "source": source,
                "ai": ai,
                "measured": measured,
                "bandwidth": bandwidth,
                "memory_roof": memory_roof,
                "attainable": attainable,
                "classification": classification,
                "symptom": symptom,
                "hypothesis": hypothesis,
                "next_experiment": next_experiment,
                "status": status,
            })

    report_path = Path("bottleneck_report.md")

    with report_path.open("w", newline="\n") as handle:
        handle.write("# Bottleneck Analysis Report\n\n")
        handle.write(f"Memory bandwidth ceiling: {MEMORY_BANDWIDTH_GB_S:.1f} GB/s\n\n")
        handle.write(f"FP32 compute ceiling: {FP32_PEAK_GFLOP_S:.1f} GFLOP/s\n\n")
        handle.write("| Kernel | Source | AI | GFLOP/s | GB/s | "
                     "Attainable roof | Roof read | Hypothesis | "
                     "Next experiment | Status |\n")
        handle.write("| ------ | ------ | -: | ------: | ---: | "
                     "--------------: | --------- | ---------- | "
                     "--------------- | ------ |\n")

        for row in rows:
            handle.write(
                f"| {row['kernel']} | {row['source']} | "
                f"{format_number(row['ai'])} | "
                f"{format_number(row['measured'])} | "
                f"{format_number(row['bandwidth'])} | "
                f"{format_number(row['attainable'])} | "
                f"{row['classification']} | "
                f"{row['hypothesis']} | "
                f"{row['next_experiment']} | "
                f"{row['status']} |\n"
            )

        handle.write("\n## Notes To Fill\n\n")
        handle.write("- Which hypothesis is strongest?\n")
        handle.write("- Which row needs a better experiment?\n")
        handle.write("- Which kernel is below the simple Roofline estimate?\n")

    print(f"Wrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
