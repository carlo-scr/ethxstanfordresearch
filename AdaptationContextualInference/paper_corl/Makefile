.PHONY: all clean
SOURCES := main.tex supplement.tex macros.tex references.bib $(wildcard sections/*.tex figures/*.tex tables/*.tex figures/data/*.tex)

all: main.pdf supplement.pdf

# The two documents reference each other (xr-hyper), so they are built together.
main.pdf supplement.pdf: $(SOURCES)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
	bibtex main
	bibtex supplement
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error supplement.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out main.pdf supplement.pdf
