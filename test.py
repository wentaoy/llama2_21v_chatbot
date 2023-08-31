import json

def prompt2json(prompt_input):
    max_length = 200
    temperature = 0.6
    top_p = 0.9
    do_sample = True
    max_new_tokens = 200
    data = {
        "input_data": {
            "input_string": [
                {
                    "role": "user",
                    "content": prompt_input
                }
            ],
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens
            }
        }
    }

    json_string = json.dumps(data, indent=2)
    return json_string

print(prompt2json("what is the history of china"))
