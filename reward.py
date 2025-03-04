def simple_reward(response, reference_answer):
    # Replace with LLM-as-judge or custom logic
    return 1.0 if response.strip() == reference_answer.strip() else 0.0
