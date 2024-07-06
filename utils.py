def get_prompt_template(template_path: str):
    with open(template_path, "r", encoding="utf-8") as file:
        template_text = file.read()
    return template_text
