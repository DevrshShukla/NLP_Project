import uuid
from typing import List, Dict, Optional, Set
from backend.nlp.topic_ranker import ScoredTopic
from backend.models.schemas import QuestionItem

class QuestionGenerator:
    @classmethod
    def generate_questions(
        cls,
        topics: List[ScoredTopic],
        parent_map: Dict[str, Optional[str]],
        children_map: Dict[str, List[str]],
        definitions_map: Dict[str, Optional[str]],
        only_high: bool = False
    ) -> List[QuestionItem]:
        questions: List[QuestionItem] = []
        topic_lookup = {t.candidate.key: t for t in topics}

        target_topics = [t for t in topics if (t.priority == "HIGH" or not only_high)]
        if not target_topics:
            target_topics = topics[:min(6, len(topics))]

        for t in target_topics:
            k = t.candidate.key
            name = t.candidate.surface
            def_sent = definitions_map.get(k)
            parent_key = parent_map.get(k)
            parent_name = topic_lookup[parent_key].candidate.surface if (parent_key and parent_key in topic_lookup) else None
            children_keys = children_map.get(k, [])
            
            # Find siblings
            siblings: List[str] = []
            if parent_key:
                siblings = [c for c in children_map.get(parent_key, []) if c != k]

            # 1. Definition question
            if def_sent or t.priority == "HIGH":
                questions.append(QuestionItem(
                    id=f"q_{uuid.uuid4().hex[:8]}",
                    topic_id=k,
                    topic_name=name,
                    question_text=f"What is {name}?",
                    question_type="definition",
                    difficulty="easy",
                    generation_method="template",
                    source_sentence=def_sent
                ))

            # 2. Explanation question
            questions.append(QuestionItem(
                id=f"q_{uuid.uuid4().hex[:8]}",
                topic_id=k,
                topic_name=name,
                question_text=f"Explain the concept of {name} and discuss its key characteristics with an example.",
                question_type="explanation",
                difficulty="medium",
                generation_method="template",
                source_sentence=def_sent
            ))

            # 3. List question (if has >= 2 children)
            if len(children_keys) >= 2:
                child_names = [topic_lookup[ck].candidate.surface for ck in children_keys if ck in topic_lookup]
                questions.append(QuestionItem(
                    id=f"q_{uuid.uuid4().hex[:8]}",
                    topic_id=k,
                    topic_name=name,
                    question_text=f"List and describe the different types or components of {name}.",
                    question_type="list",
                    difficulty="medium",
                    generation_method="template",
                    source_sentence=None
                ))

            # 4. Comparison question (if has a sibling)
            if siblings:
                sibling_k = siblings[0]
                if sibling_k in topic_lookup:
                    sibling_name = topic_lookup[sibling_k].candidate.surface
                    questions.append(QuestionItem(
                        id=f"q_{uuid.uuid4().hex[:8]}",
                        topic_id=k,
                        topic_name=name,
                        question_text=f"Differentiate between {name} and {sibling_name}.",
                        question_type="comparison",
                        difficulty="hard",
                        generation_method="template",
                        source_sentence=None
                    ))

            # 5. Why/How question
            context_domain = parent_name if parent_name else "this subject"
            questions.append(QuestionItem(
                id=f"q_{uuid.uuid4().hex[:8]}",
                topic_id=k,
                topic_name=name,
                question_text=f"Why is {name} important in {context_domain}, and how does it function?",
                question_type="why_how",
                difficulty="hard",
                generation_method="template",
                source_sentence=None
            ))

            # 6. Application question
            questions.append(QuestionItem(
                id=f"q_{uuid.uuid4().hex[:8]}",
                topic_id=k,
                topic_name=name,
                question_text=f"Where and in what scenarios is {name} applied in real-world systems?",
                question_type="application",
                difficulty="medium",
                generation_method="template",
                source_sentence=None
            ))

        return questions
