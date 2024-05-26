import streamlit as st
from ScreenWriter import ScreenWriter

st.set_page_config(page_title="LLM robot's ScreenWriter")

model = ScreenWriter()

with st.sidebar:
    st.title("LLM robot's ScreenWriter")

    st.subheader('Parameters')
    selected_robot = st.sidebar.selectbox('Choose a robot', ['Franka Emika Panda'], key='selected_robot')
    if selected_robot == 'Llama2-7B':
        possible_actions_path = 'panda_possible_actions'

    image_path = st.sidebar.text_input('Write the input image path')
    relevant_doc_path = st.sidebar.text_input('Write the relevant doc path')

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Please choose the robot and input parameters in the sidebar"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Writing scenario..."):
            response = model.predict(image_path, relevant_doc_path, prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
