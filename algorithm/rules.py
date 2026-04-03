# -*- coding: utf-8 -*-

HIGH_RISK_RULE_WORDS = ['活不下去', '想死', '自杀', '结束生命', '绝望', '崩溃', '轻生']
NEGATIVE_WORDS = ['难受', '痛苦', '孤独', '焦虑', '无助', '失眠', '压抑']


def predict_rules(text, enabled=True):
    if not enabled:
        return {'enabled': False, 'available': False, 'score': 0.0, 'hit_words': []}
    hit_words = []
    score = 0.0
    for w in HIGH_RISK_RULE_WORDS:
        if w in (text or ''):
            hit_words.append(w)
            score += 0.25
    for w in NEGATIVE_WORDS:
        if w in (text or ''):
            hit_words.append(w)
            score += 0.08
    return {
        'enabled': True,
        'available': True,
        'score': min(score, 1.0),
        'hit_words': sorted(set(hit_words)),
    }
