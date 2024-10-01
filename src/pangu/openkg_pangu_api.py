import json
import os
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from sentence_transformers import SentenceTransformer

from src.pangu.environment.examples.KB.OpenKGEnv import OpenKGEnv, lisp_to_sparql
from src.pangu.language.openkg_language import OpenKGLanguage
from src.pangu.language.plan_wrapper import Plan
from src.pangu.openkg_agent import OpenKGAgent
from src.pangu.query_util import execute_query_with_virtuoso, lisp_to_repr
from src.pangu.retrieval_api import SentenceTransformerRetriever


def format_candidates(plans):
    # A list of letters from 'a' to 't' (for at most 20 candidates)
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', "l", "m", "n", "o", "p", "q", "r", "s", "t"]
    # Using a list comprehension to format the output
    formatted_choices = ["{}. {}".format(letters[i], plan_str) for i, plan_str in enumerate(plans)]
    used_letters = letters[:len(plans)]

    # Combining the formatted choices into a single string
    formatted_string = "Candidate actions:\n" + "\n".join(formatted_choices)
    return formatted_string


def score_pairs_chat(question: str, plans, demo_retriever, demos, llm: ChatOpenAI, beam_size=5):
    if demo_retriever is not None and demos is not None:
        demo_questions = demo_retriever.get_top_k_sentences(question, 5, distinct=True)
        retrieved_demos = []
        for q in demo_questions:
            for d in demos:
                if d['question'] == q:
                    retrieved_demos.append(d)
                    break

        demo_str = '\n'.join(
            [f"Question: {demo['question']}\nLogical form: {demo['s-expression_str']}" for demo in retrieved_demos])
        system_instruction = f"You're good at understand logical forms given natural language input. Here are some examples of questions and their corresponding logical forms:\n\n{demo_str}\n\nGiven a new question, choose a candidate that is the most close to its corresponding logical form. Please only **output a single-letter option**, e.g., `a`"
    else:
        system_instruction = ("You're good at understand logical forms given natural language input. "
                              "Given a new question, choose a candidate that is the most close to its corresponding logical form. Please only **output a single-letter option**, e.g., `a`")
    demo_input1 = ["Question: Which programs are partnered with organizations of the academic type?",
                   "Candidate actions:",
                   "a. (JOIN [partner organization] (JOIN [organization type] [Industry]))",
                   "b. (JOIN [partner organization] (JOIN [organization type] [Academic]))",
                   "c. (JOIN [parent organization] (JOIN [organization type] [Academic]))",
                   "d. (JOIN [lead organization] (JOIN [organization type] [Academic]))",
                   "e. (JOIN [organization type] [Academic]))",
                   "Choice: "]
    demo_output1 = 'b'

    system_message_prompt = SystemMessage(system_instruction)
    human_message_prompt1 = HumanMessage('\n'.join(demo_input1))
    human_message_prompt2 = HumanMessagePromptTemplate.from_template(f"Question: {{question}}\n{{format_candidates}}\nChoice: ")

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt1, AIMessagePromptTemplate.from_template(demo_output1), human_message_prompt2])

    prompt_value = chat_prompt.format_prompt(question=question, format_candidates=format_candidates(plans))

    completion = llm.invoke(prompt_value.to_messages(), max_tokens=1, seed=1, logprobs=True, top_logprobs=20,
                            logit_bias={token_id: 99 for token_id in range(64, 64 + 26)})
    top_logprobs = completion.response_metadata['logprobs']['content'][0]['top_logprobs']
    top_scores = {}
    for top_log in top_logprobs:
        try:
            top_scores[plans[ord(top_log['token'].lower()) - 97]] = top_log['logprob']
        except Exception as e:
            pass
            # print('score_pairs_chat exception', e, 'TopLogprob token:', top_log['token'])
        if len(top_scores) >= beam_size:
            break

    if len(top_scores) == 0:
        print('No valid scores given by the LLM')
    return top_scores


class PanguForOpenKG:

    def __init__(self, llm_name: str = 'gpt-4o-mini', retriever='sentence-transformers/gtr-t5-base', node_to_mid_path=None, mid_to_node_path=None):
        language = OpenKGLanguage()
        environment = OpenKGEnv()
        self.symbolic_agent = OpenKGAgent(language, environment, find_new_elements=True)

        self.llm_name = llm_name
        self.node_to_mid = json.load(open(node_to_mid_path, 'r'))
        self.mid_to_node = json.load(open(mid_to_node_path, 'r'))

        # load entity/literal retriever
        if retriever.startswith('sentence-transformers/'):
            self.retrieval_model = SentenceTransformer(retriever)
            self.node_retriever = SentenceTransformerRetriever(list(self.node_to_mid.keys()), model=self.retrieval_model)
        else:
            raise NotImplementedError(f'Retriever {retriever} is not implemented')

        self.prefixes = {':': 'https://github.com/OSU-NLP-Group/HippoRAG/'}

    def text_to_query(self, question: str, top_k: int = 10, max_steps: int = 3, verbose: bool = False, openai_api_key: str = None, beam_size=5):
        """

        :param question: natural language question
        :param top_k: the number of results
        :param max_steps: the max number of beam search steps
        :param verbose: for debugging
        :return: a list of plans
        """
        if openai_api_key is not None:
            assert openai_api_key.startswith("sk-")
            os.environ['OPENAI_API_KEY'] = openai_api_key
        assert os.environ['OPENAI_API_KEY'] is not None, 'Please set OPENAI_API_KEY in environment variable'

        llm = ChatOpenAI(model=self.llm_name, temperature=0, max_retries=5, timeout=60)
        # from langchain.globals import set_llm_cache
        # from langchain_community.cache import SQLiteCache
        # set_llm_cache(SQLiteCache(database_path="exp/.langchain.db"))  # doesn't support metadata for now

        # linking
        linked_node_labels = self.node_retriever.get_top_k_sentences(question, 50, distinct=True)
        linked_node_ids = []
        for e in linked_node_labels:
            linked_node_ids.append(self.node_to_mid.get(e, None))

        # initialize plans and start beam search
        init_plans = {'Nodes': set()}
        for e in linked_node_ids[:3]:
            init_plans['Nodes'].add(Plan(e, self.mid_to_node.get(e, None)))
        self.symbolic_agent.initialize_plans(init_plans)

        cur_step = 1  # 1 to max_steps
        final_step = cur_step  # final step maybe less than the max cur_step because of determination strategy
        searched_plans = defaultdict(list)  # {step (int, starting from 1): [plan objects]}
        beams = []  # {step (int, starting from 1): [beam objects]}
        while cur_step <= max_steps:
            new_plans = self.symbolic_agent.propose_new_plans(use_all_previous=True)
            if len(new_plans) == 0:
                final_step = cur_step - 1
                break

            # filter plans using retrieval model:
            cur_plans = []
            for rtn_type in new_plans:
                # convert plan to plan_str
                for p in new_plans[rtn_type]:
                    p.plan_str = lisp_to_repr(p.plan, self.mid_to_node, self.prefixes)
                cur_plans.extend(new_plans[rtn_type])

            # recall using SentenceTransformerRetriever
            plan_recall_top_k = 20
            if len(cur_plans) > plan_recall_top_k:
                plan_ranker = SentenceTransformerRetriever([p.plan_str for p in cur_plans], model=self.retrieval_model)
                top_plans = plan_ranker.get_top_k_sentences(question, 20, distinct=True)
            else:
                top_plans = [p.plan_str for p in cur_plans]

            # ranking using LLM
            plan_str_to_scores = score_pairs_chat(question, top_plans, None, None, llm, beam_size=beam_size)
            # add plans from ranked top cur_plans to searched_plans
            for plan_str in plan_str_to_scores:
                for plan in cur_plans:
                    if plan.plan_str == plan_str:
                        plan.score = plan_str_to_scores[plan_str]
                        searched_plans[cur_step].append(plan)
                        break

            if cur_step > 1:
                stop_in_this_step = False
                # check if there exists one plan in the last step that scores higher than all the plans in this step
                for last_plan in searched_plans[cur_step - 1]:
                    last_plan_score = last_plan.score
                    if all([last_plan_score > p.score for p in searched_plans[cur_step]]):
                        stop_in_this_step = True
                        break
                if stop_in_this_step:
                    final_step = cur_step - 1
                    break  # stop searching

            # put top ranked plans back into a plan dict
            filtered_plans = defaultdict(set)
            for plan_str in plan_str_to_scores:
                for plan in cur_plans:
                    if plan.plan_str == plan_str:
                        filtered_plans[plan.rtn_type].add(plan)
                        break
            self.symbolic_agent.update_current_plans(filtered_plans)
            beams.append([str(p) for p in cur_plans])
            cur_step += 1
            final_step = cur_step

        # get plans <= final_step
        final_plans = []
        for i in range(1, final_step + 1):
            final_plans.extend(searched_plans[i])
        # rank final_plans with their scores
        final_plans = sorted(final_plans, key=lambda x: x.score, reverse=True)

        if verbose:
            print('Question:', question)
            print('Predicted entities:', linked_node_labels[:3], linked_node_ids[:3])
            print('Valid step:', final_step, 'Total step:', cur_step)

        # convert plans to SPARQL queries and get their execution results
        res = []
        num_valid = 0
        for plan in final_plans[:30]:
            sparql = lisp_to_sparql(plan.plan)
            rows = execute_query_with_virtuoso(sparql)
            if len(rows) > 0:
                num_valid += 1

            labels = []  # get labels if the results are entities
            if len(rows):
                for item in rows:
                    if len(item) == 0:
                        labels.append('')
                        continue
                    if item[0] in self.mid_to_node:
                        labels.append(self.mid_to_node[item[0]])
                    else:
                        labels.append(item[0])
            assert len(rows) == len(labels)
            res.append(
                {'input': question, 's-expression': plan.plan, 's-expression_repr': plan.plan_str, 'score': plan.score,
                 'sparql': sparql, 'results': rows, 'labels': labels, 'final_step': final_step})

            if num_valid >= top_k:
                break

        res = res[:top_k]
        if num_valid:
            res = [r for r in res if len(r['results']) > 0]
        return res, beams

    def retrieve_node(self, question: str, top_k: int = 10, distinct: bool = True):
        return self.node_retriever.get_top_k_sentences(question, top_k, distinct)
