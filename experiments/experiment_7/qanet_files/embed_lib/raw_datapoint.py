class RawDatapoint:
    def __init__(self):
        self.context_spans = []
        self.context_tokens = []
        self.context_chars = []
        self.question_tokens = []
        self.question_chars = []
        self.answer_start_idxs = []
        self.answer_end_idxs = []
        self.question_id = []

        self.context = []
        self.answer_texts = []
        self.uuid = []