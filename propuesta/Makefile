informe:
	@# Twice for ToC/References
	docker run --rm --user="`id -u`:`id -g`" -v `pwd`:/data mingc/latex pdflatex propuesta.tex
	docker run --rm --user="`id -u`:`id -g`" -v `pwd`:/data mingc/latex pdflatex propuesta.tex
.PHONY: informe
