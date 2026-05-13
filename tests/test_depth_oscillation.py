import numpy as np

from depth_oscillation import (
    count_words_in_comment,
    calculate_word_count_penalty,
    ThreadContourExtractor,
    box_counting_fractal_dimension,
)


def test_count_words_in_comment():
    assert count_words_in_comment("") == 0
    assert count_words_in_comment("Short reply.") == 2
    assert count_words_in_comment("This is a longer reply.") == 5


def test_word_count_penalty():
    assert calculate_word_count_penalty(50) == 1.0
    assert calculate_word_count_penalty(0) == 0.4
    assert calculate_word_count_penalty(20) == 0.7

    penalty_10 = calculate_word_count_penalty(10)
    penalty_30 = calculate_word_count_penalty(30)

    assert 0.4 < penalty_10 < 0.7
    assert 0.7 < penalty_30 < 1.0


def test_depth_oscillation_contour_shape():
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
    contour = extractor.extract_depth_oscillation_contour()

    assert isinstance(contour, np.ndarray)
    assert contour.shape[1] == 2
    assert len(contour) > 0


def test_contour_uses_sequential_positions():
    toy_conversation = {
        "post_title": "Example post",
        "comments": [
            {
                "body": "This is a top level reply.",
                "depth": 1,
                "replies": [
                    {
                        "body": "This is a nested reply.",
                        "depth": 2,
                        "replies": []
                    }
                ]
            }
        ]
    }

    extractor = ThreadContourExtractor(toy_conversation)
    contour = extractor.extract_depth_oscillation_contour()

    y_values = contour[:, 1]

    # Y-values should increase as traversal proceeds
    assert np.all(np.diff(y_values) >= 0)


def test_fractal_dimension_estimator_runs():
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

    fd, box_sizes, counts = box_counting_fractal_dimension(
        contour,
        save_path=None
    )

    assert isinstance(fd, float)
    assert len(box_sizes) == 15
    assert len(counts) == 15
    assert np.all(counts > 0)
