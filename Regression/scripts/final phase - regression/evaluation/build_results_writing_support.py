#!/usr/bin/env python3
from __future__ import annotations

import argparse

import pandas as pd

from claim_safety_utils import manuscript_paths, write_json, write_markdown


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description='Build manuscript writing-support files grounded in saved evidence.')
    ap.add_argument('--track', required=True, choices=['smears', 'biopsy'])
    return ap.parse_args()


def _payload(track: str) -> dict:
    mp = manuscript_paths(track)
    claim_df = pd.read_csv(mp.eval_dir / 'claim_evidence_matrix.csv', low_memory=False)

    def row(claim_id: str) -> dict:
        rec = claim_df[claim_df['claim_id'].astype(str) == claim_id].iloc[0]
        return rec.to_dict()

    payload = {
        'track': track,
        'abstract': [
            {
                'topic': 'overall_accuracy',
                'safe_wording': row('C1')['allowed_wording'],
                'unsafe_wording': row('C1')['forbidden_wording'],
                'supporting_files': row('C1')['primary_evidence_files'],
                'scope_restrictions': row('C1')['required_scope'],
                'must_mention': 'Report the weighted result together with the macro caveat and domain imbalance.',
            },
            {
                'topic': 'directional_safety',
                'safe_wording': row('C2')['allowed_wording'],
                'unsafe_wording': row('C2')['forbidden_wording'],
                'supporting_files': row('C2')['primary_evidence_files'],
                'scope_restrictions': row('C2')['required_scope'],
                'must_mention': 'State that catastrophic wrong-direction failures are rare, not absent.',
            },
        ],
        'results': [
            {
                'topic': 'domain_generalization',
                'safe_wording': row('C3')['allowed_wording'],
                'unsafe_wording': row('C3')['forbidden_wording'],
                'supporting_files': row('C3')['primary_evidence_files'],
                'scope_restrictions': row('C3')['required_scope'],
                'must_mention': 'Explicitly mention WBC dominance and include macro or domain-gap reporting.',
            },
            {
                'topic': 'roi_domain_gap',
                'safe_wording': row('C4')['allowed_wording'],
                'unsafe_wording': row('C4')['forbidden_wording'],
                'supporting_files': row('C4')['primary_evidence_files'],
                'scope_restrictions': row('C4')['required_scope'],
                'must_mention': 'Say that the ROI effect is strongest in cross-domain consistency, not dramatic pooled-MAE gain.',
            },
            {
                'topic': 'classical_baseline',
                'safe_wording': row('C5')['allowed_wording'],
                'unsafe_wording': row('C5')['forbidden_wording'],
                'supporting_files': row('C5')['primary_evidence_files'],
                'scope_restrictions': row('C5')['required_scope'],
                'must_mention': 'Keep the claim tied to the tested smear-track setup.',
            },
            {
                'topic': 'direct_regression_tradeoff',
                'safe_wording': row('C6')['allowed_wording'],
                'unsafe_wording': row('C6')['forbidden_wording'],
                'supporting_files': row('C6')['primary_evidence_files'],
                'scope_restrictions': row('C6')['required_scope'],
                'must_mention': 'State explicitly that the direct model is stronger on paired-subset MAE while the proposed model has lower wrong-direction risk.',
            },
            {
                'topic': 'input_probe_limits',
                'safe_wording': row('C7')['allowed_wording'],
                'unsafe_wording': row('C7')['forbidden_wording'],
                'supporting_files': row('C7')['primary_evidence_files'],
                'scope_restrictions': row('C7')['required_scope'],
                'must_mention': 'Label center-crop and full-field rows as inference-only probe and proxy respectively.',
            },
        ],
        'discussion': [
            {
                'topic': 'limitations',
                'safe_wording': 'The strongest performance was observed on WBC, whereas PBS and BMA remained harder, so the pooled smear result should not be interpreted as uniform domain generalization.',
                'unsafe_wording': 'The model generalizes equally well across all smear domains.',
                'supporting_files': row('C3')['primary_evidence_files'],
                'scope_restrictions': 'dataset-wise and macro reporting',
                'must_mention': 'Mention domain imbalance, macro-vs-weighted difference, and the proxy nature of the full-field baseline.',
            },
            {
                'topic': 'near_focus',
                'safe_wording': row('C9')['allowed_wording'],
                'unsafe_wording': row('C9')['forbidden_wording'],
                'supporting_files': row('C9')['primary_evidence_files'],
                'scope_restrictions': row('C9')['required_scope'],
                'must_mention': 'Near focus remains hard even when overall directional failure is low.',
            },
        ],
        'conclusion': [
            {
                'topic': 'claim_safe_conclusion',
                'safe_wording': 'Overall, the smear-track results support a strong and low-risk autofocus pipeline, with the clearest ROI-selection benefit appearing in reduced cross-domain gap rather than a large pooled-MAE gain.',
                'unsafe_wording': 'The proposed approach outperformed every baseline on every metric.',
                'supporting_files': '; '.join([row('C1')['primary_evidence_files'], row('C2')['primary_evidence_files'], row('C4')['primary_evidence_files']]),
                'scope_restrictions': 'end-to-end pooled evidence plus ROI domain-gap framing',
                'must_mention': 'Do not drop the direct-regression caveat from the conclusion if that comparison is discussed.',
            }
        ],
    }
    return payload


def _md_from_payload(payload: dict) -> str:
    lines = [f"# Results Writing Support: {payload['track']}", '']
    for section in ['abstract', 'results', 'discussion', 'conclusion']:
        lines.extend([f'## {section.title()}', ''])
        for item in payload[section]:
            lines.extend([
                f"### {item['topic']}",
                '',
                f"Safe wording: {item['safe_wording']}",
                '',
                f"Unsafe wording: {item['unsafe_wording']}",
                '',
                f"Supporting files: {item['supporting_files']}",
                '',
                f"Scope restrictions: {item['scope_restrictions']}",
                '',
                f"Must mention: {item['must_mention']}",
                '',
            ])
    return '\n'.join(lines)


def _discussion_guardrails(payload: dict) -> str:
    items = payload['discussion'] + payload['results']
    lines = ['# Discussion Guardrails', '', 'Use these guardrails to keep the discussion tied to the evidence.']
    for item in items:
        lines.extend(['', f"## {item['topic']}", '', f"Safe: {item['safe_wording']}", '', f"Unsafe: {item['unsafe_wording']}", '', f"Must mention: {item['must_mention']}"])
    return '\n'.join(lines)


def _abstract_guardrails(payload: dict) -> str:
    items = payload['abstract']
    lines = ['# Abstract Guardrails', '', 'Keep the abstract compressed and claim-safe.']
    for item in items:
        lines.extend(['', f"## {item['topic']}", '', f"Safe: {item['safe_wording']}", '', f"Unsafe: {item['unsafe_wording']}", '', f"Must mention: {item['must_mention']}"])
    return '\n'.join(lines)


def main() -> None:
    args = parse_args()
    mp = manuscript_paths(args.track)
    payload = _payload(args.track)
    write_json(mp.eval_dir / 'results_writing_support.json', payload)
    write_markdown(mp.eval_dir / 'results_writing_support.md', _md_from_payload(payload))
    write_markdown(mp.eval_dir / 'discussion_guardrails.md', _discussion_guardrails(payload))
    write_markdown(mp.eval_dir / 'abstract_guardrails.md', _abstract_guardrails(payload))


if __name__ == '__main__':
    main()
