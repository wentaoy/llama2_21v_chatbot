import streamlit as st
import urllib.request
import json
import os

# App title
st.set_page_config(page_title="🦙💬 Llama 2 On 21V Azure Chatbot")
with st.sidebar:
    st.title('Llama2🦙 on :blue[Azure 21v] :sunglasses:')

# Get Endpoint API Key Credentials
ENDPOINT_API_KEY = st.secrets["endpoint_api_key"]

# Creates an initial session state to store the LLM generated response as part of the chat message history.
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

# Display or clear chat messages
# messages (via st.chat_message()) from the chat history by iterating through the 
# messages variable in the session state.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# creates a Clear Chat History button in the sidebar, allowing users to clear the chat history by 
# leveraging the callback function
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)



# Function for generating LLaMA2 response
def generate_llama2_response(prompt_input):
    string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
    prompt_list = []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\\n\\n"
            prompt_list.append({"role": "user", "content": dict_message["content"]})
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\\n\\n"
            prompt_list.append({"role": "assistant", "content": dict_message["content"]})
    print(prompt_list)
    # reformat the input string to json
    json_input = prompt2json(prompt_input)
    # get output from model endpoint
    output = endpoint_response(json_input)
    # get string type of output instead of class bytes
    output_json = json.loads(output.decode('utf8'))
    output_string = output_json['output']
    return output_string

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
    return data

def prompt_list2json(prompt_input_list):
    max_length = 200
    temperature = 0.6
    top_p = 0.9
    do_sample = True
    max_new_tokens = 200
    data = {
        "input_data": {
            "input_string": prompt_input_list,
            "parameters": {
                "max_length": max_length,
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": do_sample,
                "max_new_tokens": max_new_tokens
            }
        }
    }
    return data

def endpoint_response(json_data):
    url = 'https://z815llama-ws-mvxsl.chinanorth3.inference.ml.azure.cn/score'
    api_key = ENDPOINT_API_KEY
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key), 'azureml-model-deployment': 'llama-2-7b-chat-hf-1' }
    body = str.encode(json.dumps(json_data))
    req = urllib.request.Request(url, body, headers)
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        return result
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))

        # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))
    
# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)