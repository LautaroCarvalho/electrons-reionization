# Build of the project outputs (declared in CLAUDE.md: `pipeline: make reproduce; clean: make clean`).
# Run from the project root. `make -j2 reproduce` runs independent figure scripts in parallel.
#
#   make reproduce      figures + numbers.json + claims.yaml + comparisons, then the provenance gate   (~2 h, -j2 ~1 h)
#   make reproduce-all  tables (provenance/data, ~65 min) first, then reproduce
#   make figures        electron_losses_IGM figure scripts (igm_*) and EM_cascades figure scripts (cas_*)
#   make numbers        provenance/numbers.json            (single writer: scripts/compute_numbers.py at the root, ~15 min)
#   make claims         provenance/claims.yaml             (single writer: scripts/build_claims.py; needs figures, numbers)
#   make comparisons    electron_losses_IGM/comparisons/   (side by side with the notebook's figures)
#   make tables         provenance/data/{F_KN_table,phi_ep_KarzasLatter,excitation_table}.json
#   make paper          paper/main.pdf from paper/main.tex (numbers via \dataref, checked by the gate)
#   make check          provenance gate: scripts/check_provenance.py
#   make test           pytest: electron_losses_IGM/tests, EM_cascades/tests, tests/ (~12 min);  make mutants (~40 min)
#   make clean          delete every generated output of `reproduce` (NOT the tables, the reference figures or inputs)

SHELL := /bin/bash
PY   ?= python3
# EM_cascades may not create files inside electron_losses_IGM/: no bytecode anywhere
export PYTHONDONTWRITEBYTECODE := 1
S    := electron_losses_IGM/scripts
SC   := EM_cascades/scripts
CFIGS := loss_fraction_rate loss_fraction_integrated ionization_yield
FIGS := gyroradius_synch distance_gyration free_streaming synch_only adiabatic_only ic_cmb ic_cmb_fixed_vs_dynamic \
        coulomb_only excitation_only ionization_only bremsstrahlung_only total_cooling comparisons accumulated phase_space
# quad's IntegrationWarnings are filtered from the log; with pipefail a failing script still fails make
# (only grep's own exit status is ignored, inside the braces).
FILTER := 2>&1 | { grep -v -iE "warn|underestim|tolerance|val, _" || true; }

.PHONY: reproduce reproduce-all figures $(addprefix fig-,$(FIGS)) $(addprefix cas-,$(CFIGS)) numbers claims comparisons \
        reference-figures tables check test mutants clean paper

reproduce: claims comparisons
	$(MAKE) paper
	$(MAKE) check

reproduce-all:
	$(MAKE) tables
	$(MAKE) reproduce

figures: $(addprefix fig-,$(FIGS)) $(addprefix cas-,$(CFIGS))

$(addprefix fig-,$(FIGS)): fig-%:
	set -o pipefail; $(PY) $(S)/fig_$*.py $(FILTER)

$(addprefix cas-,$(CFIGS)): cas-%:
	set -o pipefail; $(PY) $(SC)/fig_$*.py $(FILTER)

numbers:
	set -o pipefail; $(PY) scripts/compute_numbers.py $(FILTER)

claims: figures numbers
	$(PY) $(S)/build_claims.py

reference-figures:
	$(PY) $(S)/extract_original_figures.py

comparisons: figures reference-figures
	$(PY) $(S)/compare_with_original.py

tables:
	$(PY) $(S)/compute_fkn_table.py
	$(PY) $(S)/compute_phi_ep_table.py
	set -o pipefail; $(PY) $(S)/compute_excitation_table.py $(FILTER)

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

check:
	$(PY) scripts/check_provenance.py

test:
	$(PY) -m pytest -q -p no:cacheprovider electron_losses_IGM/tests EM_cascades/tests tests

mutants:
	$(PY) electron_losses_IGM/tests/mutants/run_mutants.py | tee electron_losses_IGM/tests/mutants/last_run.txt

clean:
	rm -f figures/igm_*.png figures/igm_*.pdf provenance/figures/igm_*.json
	rm -f figures/cas_*.png figures/cas_*.pdf provenance/figures/cas_*.json
	rm -f provenance/numbers.json provenance/claims.yaml
	rm -f electron_losses_IGM/comparisons/compare_*.png
	cd paper && latexmk -C main.tex
