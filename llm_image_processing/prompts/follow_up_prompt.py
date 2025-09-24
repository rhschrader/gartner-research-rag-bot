follow_up_prompt = """
    Given the conversation history and the user's new prompt, please determine if the new prompt is a follow-up question to the previous turn.
    A follow-up question is one that refers to or asks for more information about the immediately preceding topic.
    
    Answer with only "yes" or "no".

    ---
    Conversation History:
    {history}
    ---
    New User Prompt: {prompt}
    ---
    Is this a follow-up question?
"""