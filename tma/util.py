"""Paylaşılan küçük yardımcılar."""
import numpy as np


def wrap(a):
    """Açıyı (-pi, pi] aralığına sar."""
    return (a + np.pi) % (2 * np.pi) - np.pi
