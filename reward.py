def simple_reward(response, reference_answer):
    if not response or not reference_answer:
        return 0.0
    
    response = response.strip().lower()
    reference_answer = reference_answer.strip().lower()
    
    if response == reference_answer:
        return 1.0
    
    ref_words = set(reference_answer.split())
    resp_words = set(response.split())
    
    if not ref_words:
        return 0.0
    
    word_overlap = len(ref_words.intersection(resp_words)) / len(ref_words)
    
    len_ref = len(reference_answer)
    len_resp = len(response)
    length_ratio = min(len_ref, len_resp) / max(len_ref, len_resp) if max(len_ref, len_resp) > 0 else 0
    
    reward = 0.7 * word_overlap + 0.3 * length_ratio
    
    return reward
