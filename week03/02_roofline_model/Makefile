PYTHON ?= python3

.PHONY: analyze clean

analyze:
	$(PYTHON) roofline_from_csv.py roofline_template.csv

clean:
	rm -f roofline_report.md
