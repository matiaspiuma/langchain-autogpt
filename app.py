import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

load_dotenv()

st.title('LangChain')
st.write('A language learning app powered by OpenAI\'s GPT-3')

prompt = st.text_input('Enter a topic to write about')

title_template = PromptTemplate(
    input_variables=['topic'],
    template='Escribe un título para un video de Youtube sobre {topic}',
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='Escribe un guión para un video de youtube con el título ```{title}``` basado en esta investigación: ```{wikipedia_research}```',
)

title_memory = ConversationBufferMemory(
    input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(
    input_key='title', memory_key='chat_history')

llm = OpenAI(temperature=0.9)

title_chain = LLMChain(llm=llm, prompt=title_template,
                       verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template,
                        verbose=True, output_key='script', memory=script_memory)

# sequential_chain = SimpleSequentialChain(
#     chains=[title_chain, script_chain], verbose=True)
# sequential_chain = SequentialChain(
#     chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

wiki = WikipediaAPIWrapper()

if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    # response = sequential_chain({'topic': prompt})
    st.write(title)
    st.write(script)

    with st.expander('Título'):
        st.info(title_memory.buffer)

    with st.expander('Guión'):
        st.info(script_memory.buffer)

    with st.expander('Investigación de Wikipedia'):
        st.info(wiki_research)
