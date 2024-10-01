from crewai import Agent
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# How do I ensure ill get 100 documents
# How can I ensure that it will write proper references in the text
# How can I ensure that it will write proper references at the end of the document
# Need to make it RAG 1 by 1 for context window limit

load_dotenv()
# llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0.3)
llm = ChatGroq(model = "llama-3.2-11b-vision-preview", temperature = 0.3)

researcher_agent = Agent(
  role='Academic Researcher',
  goal="""
  Extract all the research papers from Google Scholar related to {topic} from the year {year} and above based
  on the research title.
  """,
  backstory="""
    You are a seasoned academic researcher with decades of experience in the field of research methodology
    and information retrieval. Your expertise lies in scouring academic databases, particularly Google Scholar, 
    to uncover the most relevant and up-to-date research papers on cutting-edge topics related to {topic}.
    Your mission is to find all relevant research papers from {year} and beyond on the topic of {topic}, ensuring
    that the research you gather is reliable, up-to-date, and highly valuable for further academic pursuits.
  """,
  tools=[], #defaults to an empty list
  llm=llm,
  verbose=True,
  allow_delegation=False,
  cache=True
)

filter_agent = Agent(
  role='Academic Filter expert',
  goal="""
  Filter all the papers you received such that only the most relevant papers related to the topic {topic} from
  the year {year} and beyond are returned.
  """,
  backstory="""
    You are an experienced and meticulous researcher with a sharp eye for relevance and quality. 
    Throughout your career, you've honed your ability to sift through vast amounts of information, quickly
    identifying whats crucial and discarding what isnt. Your job is to ensure that only
    the most valuable and relevant research papers related to the topic of {topic} from {year} and beyond 
    are returned and filtered.
  """,
  tools=[], #defaults to an empty list
  llm=llm,
  verbose=True,
  allow_delegation=False,
  cache=True
)

RAG_agent = Agent(
  role='Literature Reviewer',
  goal="""
  Conduct literature review for each research paper and write the output in a word document.
  """,
  backstory="""
  You are an experienced literature reviewer with a sharp eye for detail. You meticulously analyze each research 
  paper, extracting key insights on the methodology and results. Your goal is to create well-organized, 
  comprehensive summaries for each paper, ensuring that every aspect of the research is accurately represented. 
  After reviewing each paper, you compile the findings into a Word document that bears the same title as the paper 
  itself, providing a clear and structured format:
    **Title**
    **Methodology**
    **Results**
    Your methodical approach ensures that no critical detail is overlooked, and the resulting documents
    serve as valuable references for further study.
  """,
  tools=[], #defaults to an empty list
  llm=llm,
  verbose=True,
  allow_delegation=False,
  cache=True
)

writer_agent = Agent(
  role='Literature Review Writer',
  goal="""
  Write the literature review based on all the extracted information from the research papers 
  """,
  backstory="""

  """,
  tools=[], #defaults to an empty list
  llm=llm,
  verbose=True,
  allow_delegation=False,
  cache=True
)

QA_agent = Agent(
  role='RAG Agent',
  goal="""
  Peform retrieval augumented generation on each research paper and extract the 
  """,
  backstory="""

  """,
  tools=[], #defaults to an empty list
  llm=llm,
  verbose=True,
  allow_delegation=False,
  cache=True
)

