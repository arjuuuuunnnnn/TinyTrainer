def simple_reward(response, reference_answer):
    return 1.0 if response.strip() == reference_answer.strip() else 0.0
