import transformers # type: ignore
import torch # type: ignore
from pprint import pprint
import streamlit as st # type: ignore
from langchain_core.messages import AIMessage, HumanMessage # type: ignore

def load_model_tokenizer(repository):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        repository,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map = 'auto'
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(repository)
    return model, tokenizer


def get_response(text, model, tokenizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system_message = "You are a world class fitness instructor and gym trainer, you will give proper exercise and diet plans if asked, always answer the user in detail. Always answer in bullet points.'"
    prompt = f"system{system_message}user\n{text}\nassistant:"
    input_ids = tokenizer(prompt, return_tensors='pt', truncation=True).input_ids.to(device)
    outputs = model.generate(input_ids=input_ids, max_new_tokens=256)
    output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]

    return output.split("")[0]


st.set_page_config(page_title='Fitness Instructor', page_icon = "🏃‍♂️")

st.title("Fitness Instructor")


##Creating the chat_history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello I am hired as your Fitness Instructor. I will do my best to help you to the best of my Abilities.")
    ]


user_query = st.chat_input('Enter your Query here...')

if user_query is not None and user_query != "":
    model, tokenizer = load_model_tokenizer("AdityaLavaniya/TinyLlama-Fitness-Instructor")
    response = get_response(user_query, model, tokenizer)

    #Updating the chat_history:
    st.session_state.chat_history.append(HumanMessage(content = user_query ))
    st.session_state.chat_history.append(AIMessage(content = response))


    ##Displaying the chat_history in Application
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)

        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
