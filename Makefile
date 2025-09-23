.DEFAULT_GOAL := help

POETRY := poetry

help:
	@echo "Available targets:"
	@echo "  req            Export requirements.txt from Poetry"
req:
	$(POETRY) export -f requirements.txt -o requirements.txt --without-hashes

