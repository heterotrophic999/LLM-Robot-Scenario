import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

class ScreenWriter:
    model_path = "liuhaotian/llava-v1.6-mistral-7b"
    system_prompt = "You are the robot manipulator, you have to complete the task using only the following functions: 'grab', 'move to the box'."
    
    def __init__(self):
        self.image_file_path = None
        self.action_prompt = None
        self.relevant_doc_path = None
        self.possible_actions = None
        
        self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=self.model_path,
                model_base=None,
                model_name=get_model_name_from_path(self.model_path),
                load_4bit = True,
            )
    
    def get_context(self):
        loader = PyPDFLoader(self.relevant_doc_path)
        pages = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=120, 
            chunk_overlap=20 
        )
        pages = text_splitter.split_documents(pages)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs= {'device': 'cuda'},
            encode_kwargs={'normalize_embeddings': True}
        )

        db = FAISS.from_documents(documents=pages, embedding=embeddings)

        task = "What should I take with me on a hike in the forest?"

        docs = db.similarity_search(task)
        return docs[0].page_content.replace("\n", " ")
    
    @staticmethod
    def find_plan_items_in_output(output):
        plan = []
        while coords := re.search('[0-9].*\.', output):
            item = output[coords.span()[0]:coords.span()[1]]
            number = re.search('\d*. ', item)
            item = item[number.span()[1]:]
            plan.append(item)
            output = output[coords.span()[1]:]
        return plan

    def create_plan(self, first_output, second_output):
        first_plan = self.find_plan_items_in_output(first_output)
        second_plan = self.find_plan_items_in_output(second_output)

        plan = []
        for item in first_plan:
            if item in second_plan:
                plan.append(item)
        return plan

    @staticmethod
    def convert_plan(plan):
        converted_plan = []
        for item in plan:
            coords = re.search('\'.*\'', item).span()
            words = item.split()
            converted_plan.append((item[coords[0]+1:coords[1]-1], words[2]))

        return converted_plan

    def create_first_prompt(self):
        rag_context = self.get_context()
        context = f"PLEASE Use only this context: {rag_context}."

        prompt = f"{self.system_prompt}\n{context}\nYour task is: {self.action_prompt}\nWrite an action plan to achieve the goal so that each action is a separate item in the plan. In each point of the plan, clearly indicate which item you are using"

        return prompt

    @staticmethod
    def create_second_prompt(first_output):
        prompt = f"From this plan: {first_output}\n PLEASE delete items that use objects that are not in the image."

        return prompt

    @staticmethod
    def print_plan_in_cli(plan):
        for item in plan:
            print(item)
            
    def invoke(self, prompt):
        args = type('Args', (), {
                "model_path": self.model_path,
                "model_base": None,
#                 "model_name": get_model_name_from_path(self.model_path),
                "query": prompt,
                "load_4bit": True,
                "device": "cuda",
                "conv_mode": None,
                "image_file": self.image_file_path,
                "sep": ",",
                "temperature": 0,
                "top_p": None,
                "num_beams": 1,
                "max_new_tokens": 512
            })()
        output, _, _, _, _ = eval_model(args, self.tokenizer, self.model, self.image_processor, self.context_len)
        
        return output
    
    def run(self, image_file_path, relevant_doc_path, action_prompt):
        while True:
            self.image_file_path = input()
            self.relevant_doc_path = input()
            self.action_prompt = input()
            
            first_prompt = self.create_first_prompt()
            first_output = self.invoke(first_prompt)
            
            second_prompt = self.create_second_prompt(first_output)
            second_output = self.invoke(second_prompt)
            
            plan = self.create_plan(first_output, second_output)
            plan = self.convert_plan(plan)
            self.print_plan_in_cli(plan)
