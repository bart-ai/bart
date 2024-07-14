informe:
	@# twice for ToC/References
	docker run --rm --user="`id -u`:`id -g`" -v `pwd`:/data mingc/latex pdflatex informe.tex
	docker run --rm --user="`id -u`:`id -g`" -v `pwd`:/data mingc/latex pdflatex informe.tex
.PHONY: informe
