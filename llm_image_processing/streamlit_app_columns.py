import streamlit as st
from rag_chat import Rag_Chat

rag = Rag_Chat()

client = rag.client

st.set_page_config(page_title="Gartner Chat Prototype", page_icon="🤖", layout='wide')
st.title("🤖 Gartner Bot")
st.markdown("Please ask a question, and the Gartner Bot will look through 150+ Gartner research articles to help answer.")

col1, col2 = st.columns(2)

with col1:
    st.header("Chat")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "context" not in st.session_state:
        st.session_state.context = []

    if "citations" not in st.session_state:
        st.session_state.citations = []

    
    # Accept user input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):
                
                chat_input = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                is_follow_up = rag.is_follow_up_question(prompt, st.session_state.messages[:-1])

                # If it's not a follow-up, retrieve new context
                if not is_follow_up:
                    st.session_state.context = rag.retrieve_context(prompt)

                #response = rag_chat(prompt, st.session_state.messages, st.session_state.context)
                messages, st.session_state.citations = rag.generate_messages(prompt, st.session_state.messages, st.session_state.context) 

                chat_input += messages

                response = client.responses.create(
                    model=rag.chat_model,
                    input=messages
                )

                response = response.output_text

            st.markdown(response)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "system", "content": response})
        

with col2:
            
    if st.session_state.citations:
        st.header("📄 Source Pages")
        for citation in st.session_state.citations:
            pdf_name = citation['source']
            page = int(citation['page'])
            try:
                image_path = rag.render_pdf_page(pdf_name, page)
                st.image(image_path, caption=f"📄 {pdf_name} – Page {int(page)}")#, width=600)
            except Exception as e:
                st.warning(f"Couldn't load page image: {e}")
