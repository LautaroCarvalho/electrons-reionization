"""em_cascades: electron cascades in the IGM (ionization yield, loss fractions) built on igm_losses.

Uses igm_losses (electron_losses_IGM/src) read-only: every rate, medium and constant comes from there and from the
shared provenance/parameters.yaml. Importers must set sys.dont_write_bytecode before importing igm_losses so that
nothing is written inside electron_losses_IGM/ (task restriction; see EM_cascades/README.md).
"""
