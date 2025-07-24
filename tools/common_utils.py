#tools/common_utils.py

import base64
from io import BytesIO
import matplotlib.pyplot as plt


def fig_to_base64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

#creates a visual error message using matplotlib and returns it as a base64-encoded PNG string
def create_error_plot(message: str) -> str:
    """Create a simple matplotlib figure with an error message and return as base64 PNG."""
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.text(0.5, 0.5, message, fontsize=12, color='red', ha='center', va='center', wrap=True)
    ax.axis('off')
    return f"data:image/png;base64,{fig_to_base64(fig)}"
