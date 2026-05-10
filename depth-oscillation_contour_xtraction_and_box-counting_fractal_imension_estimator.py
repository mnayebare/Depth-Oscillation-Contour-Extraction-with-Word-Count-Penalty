"""
Depth-Oscillation Contour Extraction and Box-Counting Fractal Dimension Estimator

This script contains the core methods used to:
1. Convert a hierarchical Reddit conversation tree into a 2D contour.
2. Apply word-count-based depth penalization.
3. Estimate fractal dimension using the box-counting method.

Input format:
A Reddit conversation JSON object with a top-level key:
    "comments": [ ... ]

Each comment should contain:
    "body" or "text" or "content"
    "depth"
    "replies"  # optional list of child comments
"""

import numpy as np
import matplotlib.pyplot as plt


def count_words_in_comment(comment_text):
    """
    Count words in a comment.
    Returns 0 for empty or missing text.
    """
    if not comment_text or comment_text.strip() == "":
        return 0

    return len(comment_text.strip().split())


def calculate_word_count_penalty(word_count):
    """
    Calculate the word-count penalty factor.

    Penalty scheme:
    - 50+ words: no penalty, factor = 1.0
    - 20–49 words: interpolated penalty between 0.7 and 1.0
    - fewer than 20 words: interpolated penalty between 0.4 and 0.7

    The penalty is applied to the X-axis depth coordinate.
    """
    if word_count >= 50:
        return 1.0

    elif word_count >= 20:
        return 0.7 + (word_count - 20) * (0.3 / 30)

    else:
        return 0.4 + (word_count / 20) * 0.3


class ThreadContourExtractor:
    """
    Convert a hierarchical conversation tree into a 2D depth-oscillation contour.
    """

    def __init__(self, json_data):
        self.data = json_data
        self.contour_points = []

    def extract_depth_oscillation_contour(self):
        """
        Create a vertical contour with word-count penalty applied to indentation.

        X-axis:
            Penalized conversation depth.

        Y-axis:
            Sequential contour position during traversal.

        Notes:
            Transition points are added when moving into and out of reply branches.
            Therefore, the Y-axis should be interpreted as sequential contour position,
            not raw number of comments.
        """
        contour_points = []
        position_counter = [0]

        def traverse_vertically(comments, base_depth):
            for comment in comments:
                comment_text = (
                    comment.get("body", "")
                    or comment.get("text", "")
                    or comment.get("content", "")
                )

                word_count = count_words_in_comment(comment_text)
                penalty_factor = calculate_word_count_penalty(word_count)

                penalized_depth = base_depth * penalty_factor

                # Add point for the comment itself
                contour_points.append([penalized_depth, position_counter[0]])
                position_counter[0] += 1

                # Add transition points for reply branches
                if comment.get("replies"):
                    transition_depth = penalized_depth + 0.5 * penalty_factor

                    # Transition into deeper branch
                    contour_points.append([transition_depth, position_counter[0]])
                    position_counter[0] += 0.5

                    # Traverse child replies at increased base depth
                    traverse_vertically(comment["replies"], base_depth + 1)

                    # Transition back from deeper branch
                    contour_points.append([transition_depth, position_counter[0]])
                    position_counter[0] += 0.5

        traverse_vertically(self.data["comments"], 0)

        self.contour_points = np.array(contour_points)
        return self.contour_points

    def prepare_for_fractal_analysis(self):
        """
        Return the raw depth-oscillation contour for box-counting analysis.
        """
        contour = self.extract_depth_oscillation_contour()

        print("Raw contour extracted using depth-oscillation method:")
        print(f"- Total contour points: {len(contour)}")
        print(f"- X range: [{contour[:, 0].min():.3f}, {contour[:, 0].max():.3f}]")
        print(f"- Y range: [{contour[:, 1].min():.3f}, {contour[:, 1].max():.3f}]")

        return contour

    def visualize_contour(self, title="", save_path=None, figsize=(12, 8), dpi=300):
        """
        Visualize and optionally save the extracted contour.
        """
        contour = self.extract_depth_oscillation_contour()

        fig, ax = plt.subplots(figsize=figsize, facecolor="white")
        ax.set_facecolor("white")

        ax.plot(
            contour[:, 0],
            contour[:, 1],
            "k-",
            linewidth=2,
            alpha=0.8,
            label="Conversation Contour"
        )

        ax.scatter(
            contour[:, 0],
            contour[:, 1],
            c="black",
            s=8,
            alpha=0.6,
            zorder=5
        )

        ax.set_xlabel("Conversation Depth", fontweight="bold", fontsize=16)
        ax.set_ylabel("Sequential Contour Position", fontweight="bold", fontsize=16) #label corrected from number of comments
        ax.set_title(title, fontweight="bold", fontsize=18)

        ax.grid(True, alpha=0.3, color="black", linewidth=1)

        plt.tight_layout()

        if save_path:
            plt.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white",
                edgecolor="black"
            )

        plt.close()

        return contour


def box_counting_fractal_dimension(
    contour_points,
    box_sizes=None,
    save_path=None,
    figsize=(10, 6),
    dpi=300
):
    """
    Estimate the box-counting fractal dimension of a 2D contour.

    Parameters
    ----------
    contour_points : np.ndarray
        Array of [x, y] coordinates.

    box_sizes : np.ndarray, optional
        Box sizes used for box counting. If None, the estimator uses
        15 logarithmically spaced square box sizes from epsilon = 0.05
        to epsilon = 0.5 over the normalized unit square.

    save_path : str, optional
        If provided, saves the log-log scaling plot.

    Returns
    -------
    fractal_dimension : float
        Estimated slope of log(N(epsilon)) against log(1 / epsilon).

    box_sizes : np.ndarray
        Box sizes used.

    counts : np.ndarray
        Number of occupied boxes for each box size.

    Notes
    -----
    This estimator should be interpreted as an empirical box-counting
    estimate over a finite scaling range, not as proof of exact
    mathematical self-similarity.
    """

    contour_points = np.asarray(contour_points, dtype=float)

    if contour_points.ndim != 2 or contour_points.shape[1] != 2:
        raise ValueError("contour_points must be a 2D array with shape (n_points, 2).")

    if len(contour_points) < 2:
        raise ValueError("At least two contour points are required.")

    # Normalize contour to [0, 1] × [0, 1]
    x_min, x_max = contour_points[:, 0].min(), contour_points[:, 0].max()
    y_min, y_max = contour_points[:, 1].min(), contour_points[:, 1].max()

    x_range = x_max - x_min if x_max != x_min else 1.0
    y_range = y_max - y_min if y_max != y_min else 1.0

    x_norm = (contour_points[:, 0] - x_min) / x_range
    y_norm = (contour_points[:, 1] - y_min) / y_range

    normalized_points = np.column_stack([x_norm, y_norm])

    # Generate default box sizes
    if box_sizes is None:
        box_sizes = np.logspace(np.log10(0.05), np.log10(0.5), 15)
    else:
        box_sizes = np.asarray(box_sizes, dtype=float)

    counts = []

    for epsilon in box_sizes:
        occupied_boxes = set()

        for point in normalized_points:
            box_x = int(point[0] / epsilon)
            box_y = int(point[1] / epsilon)
            occupied_boxes.add((box_x, box_y))

        counts.append(len(occupied_boxes))

    counts = np.asarray(counts)

    # Fit log-log relationship
    log_inv_epsilon = np.log(1 / box_sizes)
    log_counts = np.log(counts)

    coeffs = np.polyfit(log_inv_epsilon, log_counts, 1)
    fractal_dimension = coeffs[0]

    # Plot log-log scaling relationship
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")

    ax.loglog(
        1 / box_sizes,
        counts,
        "ko-",
        linewidth=2,
        markersize=8,
        markerfacecolor="black",
        markeredgecolor="black",
        label="Observed Data"
    )

    ax.loglog(
        1 / box_sizes,
        np.exp(coeffs[1]) * (1 / box_sizes) ** coeffs[0],
        "k--",
        linewidth=3,
        alpha=0.8,
        label=f"Linear Fit: D = {fractal_dimension:.3f}"
    )

    ax.set_xlabel("Inverse Box Size", fontweight="bold", fontsize=16)
    ax.set_ylabel("Number of Occupied Boxes", fontweight="bold", fontsize=16)

    ax.legend(loc="upper left", fontsize=12, frameon=True)
    ax.grid(True, alpha=0.3, color="black", linewidth=1)

    plt.tight_layout()

    if save_path:
        plt.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="black"
        )

    plt.close()

    return fractal_dimension, box_sizes, counts


if __name__ == "__main__":
    """
    Minimal example using a toy conversation tree.
    Replace toy_conversation with a Reddit JSON conversation object.
    """
    toy_conversation = {
    "post_title": "Should AI systems be used in public decision-making?",
    "comments": [
        {
            "body": "I think they can be useful if there is transparency and public oversight.",
            "depth": 1,
            "replies": [
                {
                    "body": "Transparency is important, but most people still cannot audit these systems.",
                    "depth": 2,
                    "replies": [
                        {
                            "body": "That is why community review and independent audits should be required.",
                            "depth": 3,
                            "replies": []
                        }
                    ]
                }
            ]
        },
        {
            "body": "I disagree because these systems can reproduce bias.",
            "depth": 1,
            "replies": [
                {
                    "body": "Bias is a real concern, especially when training data are not representative.",
                    "depth": 2,
                    "replies": []
                }
            ]
        }
    ]
    }

    extractor = ThreadContourExtractor(toy_conversation)

    contour = extractor.prepare_for_fractal_analysis()

    extractor.visualize_contour(
        title="Example Depth-Oscillation Conversation Contour",
        save_path="example_conversation_contour.png"
    )

    fd, box_sizes, counts = box_counting_fractal_dimension(
        contour,
        save_path="example_box_counting_scaling_plot.png"
    )

    print(f"Estimated fractal dimension: {fd:.3f}")
    print("Box sizes:", box_sizes)
    print("Box counts:", counts)
