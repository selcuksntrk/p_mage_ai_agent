from langchain_openai import ChatOpenAI

def get_model_id(model, model_mapping):
    """
    Maps model names to their proper repository IDs.
    """
    if model not in model_mapping:
        raise ValueError(f"Unsupported model: {model}")

    return model_mapping[model]


def select_llm_agent(model="mistral-large-latest", api_key="", model_mapping={}):
    """
    This function selects the model.
    """
    if not api_key:
        raise ValueError("API key is required")

    try:
        if get_model_id(model, model_mapping) == "gpt-4o-mini":
            return ChatOpenAI(model=get_model_id(model, model_mapping), openai_api_key=api_key)


    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM agent: {str(e)}")