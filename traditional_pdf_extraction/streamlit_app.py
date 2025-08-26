import streamlit as st
#from gartner_bot import app_chat_response
from query.rag_chat import generate_answer
from render_pdf_page import render_pdf_page

st.set_page_config(page_title="Gartner Chat Prototype", page_icon="🤖")
st.title("🤖 Gartner Bot")
st.markdown("Please ask a question, and the Gartner Bot will look through 150+ Gartner research articles to help answer.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

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
            response, history, citations = generate_answer(prompt, st.session_state.messages)  # ✅ pass history
            print(citations)
            st.markdown(response)

             # ✅ Optional: show cited page image
            if citations:
                with st.sidebar:
                    st.markdown("### 📄 Source Pages")
                    for citation in citations:
                        #pdf_path = os.path.join("data", citation["source"])  # assuming PDFs live in ./data/
                        pdf = citation['source']
                        page_number = int(citation["page"])

                        try:
                            image_path = render_pdf_page(pdf, page_number)
                            st.image(image_path, caption=f"📄 {citation['source']} – Page {page_number}")
                        except Exception as e:
                            st.warning(f"Couldn't load page image: {e}")

        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})