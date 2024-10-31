# https://arxiv.org/pdf/2312.07401

template = {
    "template": '''
    For the following prompt to a chatbot, which response is more helpful(The
    information in the response needs to be direct, accurate, helpful, abundant
    and should help the user solve their task)?
    Query: {}
    Response A: {}
    Response B: {}
    FIRST provide a one-sentence comparison of the two responses and explain which you
    feel is more helpful. SECOND, on a new line, state only "A" or "B" to
    indicate which response is more helpful.
    Your response should use the format:
    Comparison: <one-sentence comparison and explanation>
    More helpful: <"A" or "B">
    ''',

    "text_func": lambda text: text.replace('"', ''),
    
    "regex_pattern": r'More helpful:\s*([AB])',
    
    "matched_char": 'A'}