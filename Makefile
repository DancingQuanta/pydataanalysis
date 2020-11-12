# Report name
REPORT = Python_Data_Analysis

# The bibliography file
BIB = references.bib

# Sources in the markdown folder or notebooks
MD_DIR = markdown
NB_DIR = notebooks
MD_SOURCES = $(wildcard $(MD_DIR)/*.md)
SOURCES = $(notdir $(basename $(MD_SOURCES)))

# Derived versions with output cells
FULL_NBS = full_notebooks

# Where to place the output
OUTPUT          = out

# Where to place the pdf, html, slides
MARKDOWN_TARGET = $(OUTPUT)/markdown/
WORD_TARGET     = $(OUTPUT)/word/
PDF_TARGET      = $(OUTPUT)/pdf/
NBPDF_TARGET    = $(OUTPUT)/nbpdf/

# Various derived files
NBS       = $(SOURCES:%=$(NB_DIR)/%.ipynb)
FULLS     = $(SOURCES:%=$(FULL_NBS)/%.ipynb)
MARKDOWNS = $(SOURCES:%=$(MARKDOWN_TARGET)%.md)
WORDS     = $(SOURCES:%=$(WORD_TARGET)%.docx)
TEXS      = $(SOURCES:%=$(PDF_TARGET)%.tex)
PDFS      = $(SOURCES:%=$(PDF_TARGET)%.pdf)
NBPDFS    = $(SOURCES:%=$(NBPDF_TARGET)%.pdf)

.PHONY: test
test: $(MD_SOURCES)
	@echo $^

PDF_FILES     = $(SOURCE_FILES:%.ipynb=$(PDF_TARGET)%_files)
NBPDF_FILES   = $(SOURCE_FILES:%.ipynb=$(NBPDF_TARGET)%_files)

# Report
REPORT_NOTEBOOK = $(FULL_NBS)/$(REPORT).ipynb
REPORT_MARKDOWN = $(MARKDOWN_TARGET)$(REPORT).md
REPORT_WORD = $(WORD_TARGET)$(REPORT).docx

## Tools
# Python
PYTHON = python

# The nbconvert alternative
NBCONVERT ?= jupyter nbconvert

# Notebook merger
NBMERGE ?= nbmerge

# Word
PANDOC ?= pandoc

# What we use for LaTeX: latexmk (preferred), or pdflatex
LATEX ?= latexmk
PDFLATEX ?= pdflatex
XELATEX ?= xelatex
BIBTEX ?= bibtex
LATEXMK ?= latexmk
LATEXMK_OPTS ?= -xelatex -quiet -f -interaction=nonstopmode

# Make directory
MKDIR = mkdir -p

# Run
EXECUTE_NOTEBOOK = $(NBCONVERT) --to notebook --execute --output-dir=$(FULL_NBS) --ExecutePreprocessor.timeout=1000

# Use standard Jupyter tools
CONVERT_TO_MARKDOWN = $(NBCONVERT) --to mddocx --output-dir=$(MARKDOWN_TARGET) #--log-level='DEBUG'

# For Word .docx files, we start from the markdown version
CONVERT_TO_WORD = $(PANDOC) -F pandoc-crossref -F pandoc-img-glob -F src/pandoc_svg.py

# To latex
CONVERT_TO_TEX = $(NBCONVERT) --to latex --output-dir=$(PDF_TARGET)

# Short targets
# The book is recreated after any change to any source
.PHONY: report all
all: report
report: report-nb report-md report-word

# Individual targets
.PHONY: word md markdown tex pdf
.PHONY: full-notebooks full fulls report-nb report-md report-word
md markdown: $(MARKDOWNS)
word doc docx: $(WORDS)
full-notebooks full fulls: $(FULLS)
tex: $(TEXS)
pdf: $(PDFS)
nbpdf: $(NBPDFS)
report-nb:  $(REPORT_NOTEBOOK)
report-md:  $(REPORT_MARKDOWN)
report-word:  $(REPORT_WORD)

# juptext
.PHONY: sync
sync: $(MD_SOURCES)
	jupytext --sync $^

.PHONY: edit jupyter lab notebook
# Invoke notebook and editor: `make jupyter lab`
edit notebook:
	jupyter notebook

lab:
	jupyter lab

# Help
.PHONY: help
help:
	@echo "Welcome to the Makefile!"
	@echo ""
	@echo "* make chapters (default) -> HTML and code for all chapters (notebooks)"
	@echo "* make (pdf|html|code|slides|word|markdown) -> given subcategory only"
	@echo "* make book -> entire book in PDF and HTML"
	@echo "* make all -> all inputs in all output formats"
	@echo "* make reformat -> reformat notebook Python code according to PEP8 guidelines"
	@echo "* make style -> style checker"
	@echo "* make crossref -> cross reference checker"
	@echo "* make stats -> report statistics"
	@echo "* make clean -> delete all derived files"
	@echo ""
	@echo "Created files end here:"
	@echo "* PDFs -> '$(PDF_TARGET)', HTML -> '$(HTML_TARGET)', Python code -> '$(CODE_TARGET)', Slides -> '$(SLIDES_TARGET)'"
	@echo "* Web site files -> '$(DOCS_TARGET)'"
	@echo ""
	@echo "Publish:"
	@echo "* make docs -> Create public version of current documents"
	@echo "* make beta -> Create beta version of current documents"
	@echo "* make publish-all -> Add docs to git, preparing for publication"
	@echo ""
	@echo "Settings:"
	@echo "* Use make PUBLISH=(nbconvert|nbpublish|bookbook) to choose a converter"
	@echo "  (default: automatic)"

# Run a notebook, (re)creating all output cells
$(FULL_NBS)/%.ipynb: $(NB_DIR)/%.ipynb
	$(EXECUTE_NOTEBOOK) $<

# Full notebook to markdown
$(MARKDOWN_TARGET)%.md:	$(FULL_NBS)/%.ipynb
	$(RM) -r $@ $(basename $@)_files
	$(CONVERT_TO_MARKDOWN) $<

# markdown to word rule TODO BIB
$(WORD_TARGET)%.docx:	$(MARKDOWN_TARGET)%.md
	$(MKDIR) $(dir $@)
	$(CONVERT_TO_WORD) $< -o $@

$(PDF_TARGET)%.tex:	$(FULL_NBS)/%.ipynb
	$(CONVERT_TO_TEX) $<
	@cd $(PDF_TARGET) && $(RM) $*.nbpub.log

# Use LaTeXMK
$(PDF_TARGET)%.pdf:	$(PDF_TARGET)%.tex
	@echo Running LaTeXMK...
	cd $(PDF_TARGET) && $(LATEXMK) $(LATEXMK_OPTS) $*
	@cd $(PDF_TARGET) && $(RM) $*.aux $*.bbl $*.blg $*.log $*.out $*.toc $*.frm $*.lof $*.lot $*.fls $*.fdb_latexmk $*.xdv
	@cd $(PDF_TARGET) && $(RM) -r $*_files
	@echo Created $@

# Conversion rules - entire report
$(REPORT_NOTEBOOK): $(FULLS)
	$(NBMERGE) $(FULLS) -o $@
	@echo Created $@

## Cleanup
.PHONY: clean-target clean-report
clean-target:
	$(RM) $(MARKDOWNS) $(WORDS)

clean-report:
	$(RM) $(REPORT_NOTEBOOK) $(REPORT_MARKDOWN) $(REPORT_WORD)

.PHONY: clean-full_notebooks clean-full clean-fulls clean-docs clean realclean
clean-full-notebooks clean-full clean-fulls:
	$(RM) $(FULLS)

clean: clean-target clean-fulls clean-report
	@echo "All derived files deleted"
