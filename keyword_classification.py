"""
Independent keyword-based classification check for the technical vs.
social/controversial post labels.

This is a simple, fully transparent rule-based classifier
built directly from the classification criteria already stated in the
manuscript (Section 4.1):

    Technical: "focused on details of algorithms, training methods,
                computer vision, and so on"
    Social/controversial: "race, gender, class, labor... DEI...
                immigrants, Islam, Israel, borders" (+ "job loss")

For each post title, it counts how many technical-category keywords and
how many social-category keywords appear, and assigns whichever category
has more matches. Posts with zero matches in both categories, or an exact
tie, are labeled 'unclear' and excluded from the agreement calculation
(and reported separately, so you can see how many titles the keyword
list failed to resolve).

Cohen's kappa is then computed between:
    - human original manual labels and 
    - this independent keyword-based classification as a supplementary validation check (not a replacement for, or claim of
equivalence to, human inter-annotator agreement).
"""

import sys
import os
import csv
import json
import re
import argparse
from sklearn.metrics import cohen_kappa_score

# ---------------------------------------------------------------------------
# Keyword lists -- grounded directly in the manuscript's own stated criteria.
# Edit/extend these if your criteria description is more specific than what's
# currently written in the manuscript.
# ---------------------------------------------------------------------------

TECHNICAL_KEYWORDS = [
    'algorithm', 'model', 'training', 'train', 'finetun', 'fine-tun',
    'computer vision', 'neural network', 'neural net', 'deep learning',
    'machine learning', 'dataset', 'data set', 'parameter', 'architecture',
    'gradient', 'inference', 'transformer', 'llm', 'hyperparameter',
    'benchmark', 'accuracy', 'code', 'coding', 'python', 'gpu', 'vram',
    'token', 'tokeniz', 'embedding', 'nlp', 'reinforcement learning',
    'optimizer', 'loss function', 'overfitting', 'regression', 'classifier',
    'classification', 'clustering', 'attention', 'rag', 'lora', 'qlora',
    'quantiz', 'rlhf', 'dpo', 'grpo', 'ppo', 'preference optimization',
    'preference learning', 'decoding', 'decode', 'decoder', 'rerank',
    're-rank', 'ranking', 'temperature', 'top_k', 'top_p',
    'repetition_penalty', 'frequency_penalty', 'presence_penalty', 'moe',
    'mixture of experts', 'rope', 'llama', 'mistral', 'gemma', 'vllm',
    'sglang', 'structured output', 'function calling',
    'byte pair encoding', 'bpe', 'distillation', 'context window',
    'context length', 'non-deterministic', 'deterministic',
    'network of experts', 'vector database'
]

# Subreddit flair tags common in ML-focused subreddits (e.g., r/MachineLearning):
# [D]iscussion, [R]esearch, [P]roject, [N]ews. Checked as plain substrings
# rather than \b-wrapped regex, since brackets aren't word characters.
TECHNICAL_FLAIR_TAGS = ['[d]', '[r]', '[p]', '[n]']

SOCIAL_KEYWORDS = [
    'race', 'racial', 'racism', 'racist', 'gender', 'gendered', 'sexism',
    'sexist', 'class', 'labor', 'labour', 'job loss', 'jobs',
    'unemployment', 'dei', 'diversity', 'equity', 'inclusion', 'immigrant',
    'immigration', 'islam', 'muslim', 'israel', 'palestine', 'border',
    'discrimination', 'bias', 'biased', 'inequality', 'inequalities',
    'privilege', 'marginalized', 'minority', 'minorities', 'lgbtq',
    'women', 'men', 'workers', 'union', 'wages', 'displaced', 'poverty',
    'wealth gap', 'economically obsolete', 'collective ownership',
    'port of entry', 'cbp', 'customs', 'liberal democracy', 'democracy',
    'election', 'surveillance', '4th amendment', 'fourth amendment',
    'facial recognition', 'black lives', 'black people', 'brown people',
    'black women'
]


def _build_pattern(keyword):
    """
    Multi-word phrases: exact whole-phrase match (leading + trailing \\b).
    Single-word keywords: leading \\b, plus an optional common inflectional
    suffix (plural/gerund/past-tense/agent-noun) before the trailing \\b.
    This lets 'finetun' match 'finetuning', 'embedding' match 'embeddings',
    'algorithm' match 'algorithms', etc. -- WITHOUT letting 'class' match
    inside 'classification' or 'classifier' (since '-ification'/'-ifier'
    aren't in the allowed suffix set).
    """
    escaped = re.escape(keyword)

    if ' ' in keyword:
        return r'\b' + escaped + r'\b'
    else:
        return r'\b' + escaped + r'(?:s|es|ed|ing|er|ers)?\b'


def classify_title(title, technical_keywords=None, social_keywords=None):
    """
    Returns ('technical', 'social', or 'unclear', technical_hits, social_hits)
    """
    if technical_keywords is None:
        technical_keywords = TECHNICAL_KEYWORDS
    if social_keywords is None:
        social_keywords = SOCIAL_KEYWORDS

    title_lower = title.lower()

    technical_hits = sum(
        1 for kw in technical_keywords
        if re.search(_build_pattern(kw), title_lower)
    )
    technical_hits += sum(1 for tag in TECHNICAL_FLAIR_TAGS if tag in title_lower)

    social_hits = sum(
        1 for kw in social_keywords
        if re.search(_build_pattern(kw), title_lower)
    )

    if technical_hits == 0 and social_hits == 0:
        return 'unclear', technical_hits, social_hits
    elif technical_hits == social_hits:
        return 'unclear', technical_hits, social_hits
    elif technical_hits > social_hits:
        return 'technical', technical_hits, social_hits
    else:
        return 'social', technical_hits, social_hits


def load_titles_from_json_folder(folder_path):
    """
    Loads post_id, title, and conversation_type from a json_data folder,
    using the exact same conventions as load_conversations_from_folder()
    in step3_box_counting_controversial_and_technical_posts.py:

      - post_id       = filename, minus '_reddit_comments_with_time.json'
      - conversation_type = 'social' if 'con' in post_id (lowercased),
                             'technical' if 'tec' in post_id, else 'unknown'
      - title         = data['post_title'] field inside the JSON file

    Returns a list of dicts: {'post_id', 'title', 'conversation_type'}
    """
    entries = []

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return entries

    json_files = [
        f for f in os.listdir(folder_path)
        if f.endswith('_reddit_comments_with_time.json')
    ]

    print(f"Found {len(json_files)} conversation files in '{folder_path}'")

    for json_file in json_files:
        file_path = os.path.join(folder_path, json_file)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            post_id = json_file.replace('_reddit_comments_with_time.json', '')

            if "con" in post_id.lower():
                conversation_type = "social"
            elif "tec" in post_id.lower():
                conversation_type = "technical"
            else:
                conversation_type = "unknown"

            title = data.get('post_title', None)

            if not title:
                print(f"  Skipping {post_id}: no 'post_title' field found in JSON.")
                continue

            entries.append({
                'post_id': post_id,
                'title': title,
                'conversation_type': conversation_type
            })

        except Exception as e:
            print(f"  Error loading {json_file}: {e}")
            continue

    return entries


def load_titles_from_csv(input_csv):
    """
    Loads post_id, title, and conversation_type from a CSV with those
    exact column names. Returns a list of dicts in the same shape as
    load_titles_from_json_folder().
    """
    entries = []

    with open(input_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            entries.append({
                'post_id': row['post_id'],
                'title': row['title'],
                'conversation_type': row['conversation_type']
            })

    return entries


def compute_agreement(entries, output_csv='keyword_classification_check_results.csv'):
    """
    Given a list of {'post_id', 'title', 'conversation_type'} dicts, runs
    the keyword classifier on each title, saves full per-post results, and
    computes Cohen's kappa / percent agreement against the original labels
    (excluding posts the keyword classifier could not resolve).
    """
    rows = []

    for entry in entries:
        post_id = entry['post_id']
        title = entry['title']
        original_label = entry['conversation_type'].strip().lower()

        predicted_label, tech_hits, social_hits = classify_title(title)

        rows.append({
            'post_id': post_id,
            'title': title,
            'original_label': original_label,
            'keyword_predicted_label': predicted_label,
            'technical_keyword_hits': tech_hits,
            'social_keyword_hits': social_hits,
            'agree': (predicted_label == original_label)
        })

    if not rows:
        print("No entries to evaluate.")
        return None, None, 0, 0

    # Save full per-post results for manual inspection
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Only compare against 'social'/'technical' original labels; skip
    # any 'unknown' original labels (e.g. filenames without 'con'/'tec').
    comparable_rows = [r for r in rows if r['original_label'] in ('social', 'technical')]
    skipped_unknown = len(rows) - len(comparable_rows)

    resolved_rows = [r for r in comparable_rows if r['keyword_predicted_label'] != 'unclear']
    unclear_rows = [r for r in comparable_rows if r['keyword_predicted_label'] == 'unclear']

    n_total = len(comparable_rows)
    n_resolved = len(resolved_rows)
    n_unclear = len(unclear_rows)

    print(f"\nTotal posts loaded:              {len(rows)}")
    if skipped_unknown:
        print(f"Skipped (original label unknown): {skipped_unknown}")
    print(f"Posts compared:                  {n_total}")
    print(f"Resolved by keywords:            {n_resolved} ({100 * n_resolved / n_total:.1f}%)")
    print(f"Unclear (excluded):               {n_unclear} ({100 * n_unclear / n_total:.1f}%)")

    if n_resolved > 1:
        original_labels = [r['original_label'] for r in resolved_rows]
        predicted_labels = [r['keyword_predicted_label'] for r in resolved_rows]

        percent_agreement = sum(
            1 for o, p in zip(original_labels, predicted_labels) if o == p
        ) / n_resolved

        kappa = cohen_kappa_score(original_labels, predicted_labels)

        print(f"\nPercent agreement (on resolved posts): {percent_agreement * 100:.1f}%")
        print(f"Cohen's kappa (on resolved posts):     {kappa:.3f}")
        print(f"\nFull per-post results saved to: {output_csv}")

        return kappa, percent_agreement, n_resolved, n_unclear
    else:
        print("\nNot enough resolved posts to compute Cohen's kappa.")
        return None, None, n_resolved, n_unclear


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Independent keyword-based classification check for post labels."
    )
    parser.add_argument(
        '--json-folder',
        default=None,
        help="Path to the json_data folder (uses the same convention as the main pipeline)."
    )
    parser.add_argument(
        '--csv',
        default=None,
        help="Path to a CSV with columns: post_id, title, conversation_type"
    )
    parser.add_argument(
        '--output',
        default='keyword_classification_check_results.csv',
        help="Where to save the full per-post results CSV."
    )

    args = parser.parse_args()

    if args.json_folder:
        entries = load_titles_from_json_folder(args.json_folder)
    elif args.csv:
        entries = load_titles_from_csv(args.csv)
    else:
        # Default: try 'json_data' in the current directory, matching the
        # main pipeline's default folder name.
        print("No --json-folder or --csv given; defaulting to './json_data'")
        entries = load_titles_from_json_folder('json_data')

    compute_agreement(entries, output_csv=args.output)
