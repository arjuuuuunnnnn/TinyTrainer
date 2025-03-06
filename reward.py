def simple_reward(response, reference_answer):
    """
    Calculate reward based on response similarity to reference answer.
    A more robust implementation than exact match.
    
    Args:
        response (str): Model generated response
        reference_answer (str): Expected answer
        
    Returns:
        float: Reward value between 0 and 1
    """
    # Handle None or empty values
    if not response or not reference_answer:
        return 0.0
    
    # Clean up text for comparison
    response = response.strip().lower()
    reference_answer = reference_answer.strip().lower()
    
    # Exact match gets highest reward
    if response == reference_answer:
        return 1.0
    
    # Partial content match reward
    # Calculate what percentage of words from reference are in response
    ref_words = set(reference_answer.split())
    resp_words = set(response.split())
    
    if not ref_words:  # Avoid division by zero
        return 0.0
    
    # Intersection of words / total reference words
    word_overlap = len(ref_words.intersection(resp_words)) / len(ref_words)
    
    # Length penalty - discourage responses that are too long or too short
    len_ref = len(reference_answer)
    len_resp = len(response)
    length_ratio = min(len_ref, len_resp) / max(len_ref, len_resp) if max(len_ref, len_resp) > 0 else 0
    
    # Combined reward
    reward = 0.7 * word_overlap + 0.3 * length_ratio
    
    return reward
