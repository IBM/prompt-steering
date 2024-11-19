from langchain_core.prompts import PromptTemplate
from langchain.output_parsers.structured import ResponseSchema, StructuredOutputParser
import random


def format_prompt(statement,
                  criteria,
                  template_category,
                  output_format,
                  steering_prompt=None
                  ):

    # templates
    templates = _load_templates()

    system_prompt_template_object = PromptTemplate(
        template=templates['system_prompt_template'],
        input_variables=["steering_prompt",
                         "criteria"]
    )

    if output_format:
        # output format
        response_schema = [ResponseSchema(name=field['name'], description=field['description'])
                           for field in output_format]
        output_parser = StructuredOutputParser.from_response_schemas(response_schema)

        query_template_object = PromptTemplate(
            template=templates['query_template_format'],
            input_variables=["question"],
            partial_variables={"format_instructions": output_parser.get_format_instructions()}
        )
    else:
        query_template_object = PromptTemplate(
            template=templates['query_template'],
            input_variables=["question"]
        )

    # fill templates
    criteria_block = get_criteria_block(criteria=criteria)
    steering_prompt = steering_prompt if steering_prompt else ""
    system_prompt = system_prompt_template_object.format(steering_prompt=steering_prompt,
                                                         criteria=criteria_block)

    question = f"Is the following statement something you would say?\n{statement}"
    query = query_template_object.format(question=question)

    return model_template(system_prompt=system_prompt,
                          query=query,
                          template_category=template_category)


def get_criteria_block(criteria):
    criteria_string = _string_from_dict(criteria)
    return criteria_string


def _string_from_dict(input_dict, valid_keys=None):
    if valid_keys is None:
        valid_keys = input_dict.keys()
    formatted_string = [f"{key}: {input_dict[key]}" for key in input_dict if key in valid_keys]
    return '\n'.join(formatted_string)


def _load_templates():
    system_prompt_template = """{steering_prompt}

You will be asked to provide an answer to a question. When answering the question, please use the following criteria:
{criteria}
"""

    query_template_format = """{question}

{format_instructions}

Your answer: """

    query_template = """{question}
    
Your answer: """

    return {
        "system_prompt_template": system_prompt_template,
        "query_template_format": query_template_format,
        "query_template": query_template
    }


def model_template(system_prompt, query, template_category):
    if template_category == "llama-2":
        return f"""<s>[INST] <<SYS>>\n\u007b\u007b {system_prompt} \u007d\u007d\n<</SYS>>\n\n\u007b\u007b {query} \u007d\u007d [/INST] \u007b\u007b """
    elif template_category == "llama-3":
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"""
    elif template_category == "mistral":
        return f"<s>[INST] {system_prompt}[/INST] </s>[INST] {query} [/INST] "
    elif template_category == "gpt":
        return f"""[SYSTEM]\n{system_prompt}\n\n[USER]\n{query}\n\n[END]"""
    elif template_category == "gemma":
        return f"""<start_of_turn>user\n{system_prompt}\n\n{query}<end_of_turn>\n<start_of_turn>model\n"""
    elif template_category == "granite":
        return f"""<|system|>\n{system_prompt}\n<|user|>\n{query}\n<|assistant|>\n"""
    elif template_category == "phi3":
        return f"""<|system|>\n{system_prompt}<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>\n"""
    else:
        raise ValueError(f"Unsupported prompt template category: {template_category}")
