"""Reject Japanese tokenization in this English-only alignment environment.

Qwen's aligner imports nagisa unconditionally, but this deployment has no
Python 3.12 DyNet wheel. English tokenization does not call nagisa.tagging.
"""
def tagging(*args, **kwargs):
    raise RuntimeError('Japanese tokenization is unavailable in this English-only run')
