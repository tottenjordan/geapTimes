"""Vertex AI managed-pipeline (KFP v2) orchestration for geapTimes.

This subpackage lifts the Stage 3 in-process experiment loop onto Vertex AI Pipelines and adds the
custom-container TimesFM cloud-serving path. As in earlier stages, the real logic lives in plain,
injected-seam functions (so it unit-tests offline); the KFP ``@component``/``@pipeline`` wrappers
and the serving web app are thin shells over that logic.
"""
