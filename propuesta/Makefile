propuesta:
	@# twice for ToC/References
	docker run --rm --user="`id -u`:`id -g`" -v `pwd`:/data mingc/latex pdflatex propuesta.tex
	docker run --rm --user="`id -u`:`id -g`" -v `pwd`:/data mingc/latex pdflatex propuesta.tex
.PHONY: propuesta

all: propuesta
	pdftk propuesta.pdf extradocs/cartas.pdf extradocs/cv-fede.pdf extradocs/materias-fede.pdf extradocs/cv-casi.pdf extradocs/materias-casi.pdf cat output propuesta-del-mazo-pastine.pdf
