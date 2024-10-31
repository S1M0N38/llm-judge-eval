# https://arxiv.org/pdf/2306.05685
# Figure 9: The prompt for multi-turn pairwise comparison.

template = {
    "template": '''
    [System]
    Please act as an impartial judge and evaluate the quality of the responses provided by two
    AI assistants to the user question displayed below. You should choose the assistant that
    follows the user's instructions and answers the user's question better. Your evaluation
    should consider factors such as the helpfulness, relevance, accuracy, depth, creativity,
    and level of detail of their responses. Begin your evaluation by comparing the two
    responses and provide a short explanation. Avoid any position biases and ensure that the
    order in which the responses were presented does not influence your decision. Do not allow
    the length of the responses to influence your evaluation. Do not favor certain names of
    the assistants. Be as objective as possible. After providing your explanation, output your
    final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]"
    if assistant B is better.

    <|The Start of Assistant A's Conversation with User|>
    {0}
    ### Assistant A:
    {1}
    <|The End of Assistant A's Conversation with User|>

    <|The Start of Assistant B's Conversation with User|>
    {0}
    ### Assistant B:
    {2}
    <|The End of Assistant B's Conversation with User|>
    ''',

    "text_func": lambda text: text,
    
    "regex_pattern": r'\[\[([AB])\]\]',
    
    "matched_char": 'A'}