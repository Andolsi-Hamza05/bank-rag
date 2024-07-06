def get_prompt_tempalte():
    template_file = "prompts/naive_rag.txt"
    with open(template_file, "r", encoding="utf-8") as file:
        template_text = file.read()
    return template_text
