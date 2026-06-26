"""Experiment layer: Design-of-Experiments expansion, tracking, metrics, and the run loop.

Stage 3 drives the Stage 2 forecasters: :mod:`geaptimes.experiment.doe` expands one config into a
grid of variants, :mod:`geaptimes.experiment.runner` runs each (variant x enabled model) through
:class:`~geaptimes.models.base.Forecaster`, and :mod:`geaptimes.experiment.tracking` logs params,
:mod:`geaptimes.experiment.metrics`, and artifacts to Vertex AI Experiments + GCS.
"""
