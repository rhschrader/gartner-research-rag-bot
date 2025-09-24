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

    # Display chat messages from history on app rerun
    #for message in st.session_state.messages:
        #with st.chat_message(message["role"]):
            #st.markdown(message["content"])

    citations = []
    # Accept user input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to chat history
        #st.session_state.messages.append({"role": "user", "content": prompt})
        #print(prompt)
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("system"):

            with st.spinner("Thinking..."):
                
                chat_input = [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]

                messages, citations = rag.generate_messages(prompt, chat_input) 

                chat_input += messages

                response = client.responses.create(
                    model=rag.chat_model,
                    input=chat_input
                )

                response = response.output_text

            st.markdown(response)

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "system", "content": response})
        

with col2:
            
    if citations:
        st.header("📄 Source Pages")
        for citation in citations:
            pdf_name = citation['source']
            page = int(citation['page'])
            #print(f"Document: {citation['source']}\t Page: {citation['page']}")
            #print(f"Document: {pdf_name}\t Page: {page}")
            try:
                #image_path = rag.get_pdf_image(citation)
                image_path = rag.render_pdf_page_debug(pdf_name, page)
                #st.image(image_path, caption=f"📄 {citation['source']} – Page {int(citation['page'])}")#, width=600)
                st.image(image_path, caption=f"📄 {pdf_name} – Page {int(page)}")#, width=600)
            except Exception as e:
                st.warning(f"Couldn't load page image: {e}")
