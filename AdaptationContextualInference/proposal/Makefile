.PHONY: all clean

# Freeze PDF metadata so repeated builds of identical sources are byte-stable. Override only when
# intentionally cutting a differently dated artifact.
SOURCE_DATE_EPOCH ?= 1787356800
export SOURCE_DATE_EPOCH

PAPER_SOURCES := main.tex references.bib $(wildcard sections/*.tex) $(wildcard figures/*.pdf)

all: main.pdf

main.pdf: $(PAPER_SOURCES)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
