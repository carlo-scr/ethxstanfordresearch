.PHONY: all clean

SOURCES := main.tex references.bib $(wildcard sections/*.tex)

all: main.pdf figure_plan.pdf

main.pdf: $(SOURCES)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

figure_plan.pdf: figure_plan.tex sections/figures.tex
	pdflatex -interaction=nonstopmode -halt-on-error figure_plan.tex
	pdflatex -interaction=nonstopmode -halt-on-error figure_plan.tex

clean:
	rm -f main.aux main.bbl main.blg main.log main.out figure_plan.aux figure_plan.log figure_plan.out
