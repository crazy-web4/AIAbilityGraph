class AnswerRelavency:
    question_generation_prompt = {
        "instruction": "Generate a question related to the given answer and determine if the answer is ambiguous. Mark with 1 if ambiguous, 0 if clear. Ambiguous answers are vague, evasive, or unclear responses like 'I don't know' or 'I'm not sure'.",
        "output_format_instruction": "Output should be a JSON instance conforming to the following schema:\n\nExample: For schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\n, the object {\"foo\": [\"bar\", \"baz\"]} is valid. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is invalid.\n\nOutput JSON schema:\n```\n{\"type\": \"object\", \"properties\": {\"question\": {\"title\": \"Question\", \"type\": \"string\"}, \"noncommittal\": {\"title\": \"Noncommittal\", \"type\": \"integer\"}}, \"required\": [\"question\", \"noncommittal\"]}\n```\n\nReturn only the JSON string enclosed in triple backticks without any preamble or explanation.",
        "examples": [
            {
                "answer": "Albert Einstein was born in Germany.",
                "context": "Albert Einstein was a German-born theoretical physicist widely regarded as one of the greatest and most influential scientists of all time.",
                "output": {
                    "question": "Where was Albert Einstein born?",
                    "noncommittal": 0
                }
            },
            {
                "answer": "It can change skin color based on environmental temperature.",
                "context": "A recent scientific study discovered a new species of frog in the Amazon rainforest with the unique ability to change skin color according to environmental temperature.",
                "output": {
                    "question": "What unique ability does the newly discovered frog species have?",
                    "noncommittal": 0
                }
            },
            {
                "answer": "Mount Everest",
                "context": "The highest mountain from sea level on Earth is a famous peak located in the Himalayas.",
                "output": {
                    "question": "What is the highest mountain on Earth?",
                    "noncommittal": 0
                }
            },
            {
                "answer": "I don't know about the breakthrough features of smartphones invented in 2023 because I lack information after 2022.",
                "context": "In 2023, a groundbreaking invention was announced: a smartphone with one-month battery life that revolutionized mobile technology usage.",
                "output": {
                    "question": "What was the breakthrough feature of the smartphone invented in 2023?",
                    "noncommittal": 1
                }
            }
        ]
    }

class Faithfulness:
    nli_statements_message_prompt = {
        "instruction": "Your task is to evaluate the veracity of a set of statements based on the given context. For each statement, return judgment result 1 if it can be directly inferred from the context, or 0 if no direct evidence exists in the context.",
        "output_format_instruction": "Output should be a JSON instance conforming to the following schema:\n\nExample: For schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\n, the object {\"foo\": [\"bar\", \"baz\"]} is valid. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is invalid.\n\nOutput JSON schema:\n```\n{\"type\": \"array\", \"items\": {\"$ref\": \"#/definitions/StatementFaithfulnessAnswer\"}, \"definitions\": {\"StatementFaithfulnessAnswer\": {\"title\": \"StatementFaithfulnessAnswer\", \"type\": \"object\", \"properties\": {\"statement\": {\"title\": \"Statement\", \"description\": \"the original statement, word-by-word\", \"type\": \"string\"}, \"reason\": {\"title\": \"Reason\", \"description\": \"verdict rationale\", \"type\": \"string\"}, \"verdict\": {\"title\": \"Verdict\", \"description\": \"faithfulness verdict (0/1)\", \"type\": \"integer\"}}, \"required\": [\"statement\", \"reason\", \"verdict\"]}}}\n```\n\nReturn only the JSON string enclosed in triple backticks without any preamble or explanation.",
        "examples": [
            {
                "context": "John is a student at XYZ University. He is majoring in computer science and taking several courses this semester including data structures, algorithms, and database management. John is a diligent student who spends significant time studying and completing assignments, often working late in the library on projects.",
                "statements": [
                    "John majors in biology.",
                    "John is taking an artificial intelligence course.",
                    "John is a diligent student.",
                    "John has a part-time job."
                ],
                "answer": [
                    {
                        "statement": "John majors in biology.",
                        "reason": "John's major is explicitly stated as computer science. There is no information indicating he majors in biology.",
                        "verdict": 0
                    },
                    {
                        "statement": "John is taking an artificial intelligence course.",
                        "reason": "The context mentions John's current courses but does not include artificial intelligence. Therefore, there is insufficient evidence to confirm this statement.",
                        "verdict": 0
                    },
                    {
                        "statement": "John is a diligent student.",
                        "reason": "The context states he spends significant time studying and completing assignments, and frequently works late in the library on projects - all indicators of diligence.",
                        "verdict": 1
                    },
                    {
                        "statement": "John has a part-time job.",
                        "reason": "No information is provided in the context about John having a part-time job.",
                        "verdict": 0
                    }
                ]
            },
            {
                "context": "Photosynthesis is the process by which plants, algae, and certain bacteria convert light energy into chemical energy.",
                "statements": [
                    "Albert Einstein was a genius."
                ],
                "answer": [
                    {
                        "statement": "Albert Einstein was a genius.",
                        "reason": "The statement is unrelated to the context",
                        "verdict": 0
                    }
                ]
            }
        ]
    }

    statement_prompt = {
        "instruction": "Given a question, an answer, and sentences from the answer, analyze the complexity of each sentence in 'sentences' and decompose each sentence into one or more fully understandable statements while ensuring no pronouns are used in each statement. Format the output as JSON.",
        "output_format_instruction": "Output should be a JSON instance conforming to the following schema:\n\nExample: For schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\n, the object {\"foo\": [\"bar\", \"baz\"]} is valid. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is invalid.\n\nOutput JSON schema:\n```\n{\"type\": \"array\", \"items\": {\"$ref\": \"#/definitions/Statements\"}, \"definitions\": {\"Statements\": {\"title\": \"Statements\", \"type\": \"object\", \"properties\": {\"sentence_index\": {\"title\": \"Sentence Index\", \"description\": \"Index of the sentence from the statement list\", \"type\": \"integer\"}, \"simpler_statements\": {\"title\": \"Simpler Statements\", \"description\": \"the simpler statements\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"sentence_index\", \"simpler_statements\"]}}}\n```\n\nReturn only the JSON string enclosed in triple backticks without any preamble or explanation.",
        "examples": [
            {
                "question": "Who was Albert Einstein and what was he famous for?",
                "answer": "He was a German-born theoretical physicist widely considered one of the greatest and most influential physicists of all time. He is most famous for developing the theory of relativity and also made significant contributions to the development of quantum mechanics theory.",
                "sentences": "\n        0:He was a German-born theoretical physicist, widely considered one of the greatest and most influential physicists of all time.\n        1:He is most famous for developing the theory of relativity and also made significant contributions to the development of quantum mechanics theory.\n        ",
                "analysis": [
                    {
                        "sentence_index": 0,
                        "simpler_statements": [
                            "Albert Einstein was a German-born theoretical physicist.",
                            "Albert Einstein was considered one of the greatest and most influential physicists of all time."
                        ]
                    },
                    {
                        "sentence_index": 1,
                        "simpler_statements": [
                            "Albert Einstein is most famous for developing the theory of relativity.",
                            "Albert Einstein made significant contributions to the development of quantum mechanics theory."
                        ]
                    }
                ]
            }
        ]
    }




class ContextRecall:
    context_recall_prompt = {
        "instruction": "Given a context and an answer, analyze each sentence in the answer and determine whether the sentence can be attributed to the given context. Use only 'Yes' (1) or 'No' (0) as binary classification. Output JSON with rationale.",
        "output_format_instruction": "Output should be a well-formed JSON instance conforming to the following schema:\n\nExample: For schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\n, the object {\"foo\": [\"bar\", \"baz\"]} is a valid instance. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is invalid.\n\nOutput JSON schema:\n```\n{\"type\": \"array\", \"items\": {\"$ref\": \"#/definitions/ContextRecallClassificationAnswer\"}, \"definitions\": {\"ContextRecallClassificationAnswer\": {\"title\": \"ContextRecallClassificationAnswer\", \"type\": \"object\", \"properties\": {\"statement\": {\"title\": \"Statement\", \"type\": \"string\"}, \"attributed\": {\"title\": \"Attributed\", \"type\": \"integer\"}, \"reason\": {\"title\": \"Reason\", \"type\": \"string\"}}, \"required\": [\"statement\", \"attributed\", \"reason\"]}}}\n```\n\nReturn only the JSON string enclosed in triple backticks without any preamble or explanation.",
        "examples": [
            {
                "question": "What information can you tell me about Albert Einstein?",
                "context": "Albert Einstein (March 14, 1879 - April 18, 1955) was a German-born theoretical physicist widely considered one of the greatest and most influential scientists of all time. He is best known for developing the theory of relativity and made significant contributions to quantum mechanics, making him a central figure in the revolutionary transformation of scientific understanding of nature during the early decades of 20th century physics. His mass-energy equivalence formula E=mc², derived from relativity theory, is known as 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a critical step in quantum theory development. His work also became renowned for its impact on the philosophy of science. In a 1999 survey by the British magazine 'Physics World' of 130 leading physicists worldwide, Einstein was voted 'the greatest physicist of all time'. His intellectual achievements and originality have made Einstein synonymous with genius.",
                "answer": "Albert Einstein was born on March 14, 1879, and was a German-born theoretical physicist widely considered one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his contributions to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895.",
                "classification": [
                    {
                        "statement": "Albert Einstein was born on March 14, 1879, and was a German-born theoretical physicist widely considered one of the greatest and most influential scientists of all time.",
                        "attributed": 1,
                        "reason": "Einstein's birth date is explicitly mentioned in the context."
                    },
                    {
                        "statement": "He received the 1921 Nobel Prize in Physics for his contributions to theoretical physics.",
                        "attributed": 1,
                        "reason": "The context contains an exact sentence stating this fact."
                    },
                    {
                        "statement": "He published 4 papers in 1905.",
                        "attributed": 0,
                        "reason": "No mention of the papers he wrote in the context."
                    },
                    {
                        "statement": "Einstein moved to Switzerland in 1895.",
                        "attributed": 0,
                        "reason": "No supporting evidence exists in the context."
                    }
                ]
            },
            {
                "question": "Who won the 2020 ICC Cricket World Cup?",
                "context": "The 2022 ICC Men's T20 World Cup was held in Australia from October 16 to November 13, 2022, marking the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England defeated Pakistan by five wickets in the final to claim their second ICC Men's T20 World Cup championship.",
                "answer": "England",
                "classification": [
                    {
                        "statement": "England won the 2022 ICC Men's T20 World Cup.",
                        "attributed": 1,
                        "reason": "The context clearly states England defeated Pakistan to win the World Cup."
                    }
                ]
            },
            {
                "question": "What is the primary fuel of the sun?",
                "context": "NULL",
                "answer": "Hydrogen",
                "classification": [
                    {
                        "statement": "The sun's primary fuel is hydrogen.",
                        "attributed": 0,
                        "reason": "No information is present in the context."
                    }
                ]
            }
        ]
    }


class ContextPrecision:
    context_precision_prompt = {
        "instruction": "Given a question, an answer, and a context, verify whether the context helps in deriving the given answer. Output verdict '1' if it does, '0' otherwise.",
        "output_format_instruction": "Output should be a well-formed JSON instance conforming to the following JSON schema.\n\nExample: For schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\n, the object {\"foo\": [\"bar\", \"baz\"]} is a valid schema instance. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is invalid.\n\nOutput JSON schema:\n```\n{\"description\": \"Answer for the verification task whether the context was useful.\", \"type\": \"object\", \"properties\": {\"reason\": {\"title\": \"Reason\", \"description\": \"Reason for verification\", \"type\": \"string\"}, \"verdict\": {\"title\": \"Verdict\", \"description\": \"Binary (0/1) verdict of verification\", \"type\": \"integer\"}}, \"required\": [\"reason\", \"verdict\"]}\n```\n\nReturn only the JSON string enclosed in triple backticks without any preamble or explanation.",
        "examples": [
            {
                "question": "What information can you tell me about Albert Einstein?",
                "context": "Albert Einstein (March 14, 1879 – April 18, 1955) was a German-born theoretical physicist widely considered one of the greatest and most influential scientists of all time. He is best known for developing the theory of relativity and made significant contributions to quantum mechanics, making him a central figure in the revolutionary transformation of scientific understanding of nature during the early decades of 20th century physics. His mass-energy equivalence formula E=mc², derived from relativity theory, is known as 'the world's most famous equation'. He received the 1921 Nobel Prize in Physics 'for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect', a critical step in quantum theory development. His work also became renowned for its impact on the philosophy of science. In a 1999 survey by the British magazine 'Physics World' of 130 leading physicists worldwide, Einstein was voted 'the greatest physicist of all time'. His intellectual achievements and originality have made Einstein synonymous with genius.",
                "answer": "Albert Einstein was born on March 14, 1879, and was a German-born theoretical physicist widely considered one of the greatest and most influential scientists of all time. He received the 1921 Nobel Prize in Physics for his contributions to theoretical physics. He published 4 papers in 1905. Einstein moved to Switzerland in 1895.",
                "verification": {
                    "reason": "The provided context contains key information about Albert Einstein's life and contributions that directly supports the statements in the answer.",
                    "verdict": 1
                }
            },
            {
                "question": "Who won the 2020 ICC Cricket World Cup?",
                "context": "The 2022 ICC Men's T20 World Cup was held in Australia from October 16 to November 13, 2022, marking the eighth edition of the tournament. Originally scheduled for 2020, it was postponed due to the COVID-19 pandemic. England defeated Pakistan by five wickets in the final to claim their second ICC Men's T20 World Cup championship.",
                "answer": "England",
                "verification": {
                    "reason": "The context clarifies the 2020 ICC Cricket World Cup situation and explicitly states that England won the championship originally scheduled for 2020 but held in 2022.",
                    "verdict": 1
                }
            },
            {
                "question": "What is the highest mountain in the world?",
                "context": "The Andes Mountains are the longest continental mountain range in the world, located in South America. Spanning seven countries, it contains many of the highest peaks in the Western Hemisphere. The mountain range is notable for its diverse ecosystems including high-altitude Andean plateaus and Amazon rainforest regions.",
                "answer": "Mount Everest.",
                "verification": {
                    "reason": "The provided context discusses the Andes Mountains, while the answer refers to Mount Everest which is not mentioned or related to the world's highest mountain question in the context.",
                    "verdict": 0
                }
            }
        ]
    }


class AnswerCorrectness:
    correctness_prompt = {
        "instruction": "Given ground_truth and decomposed answer statements, analyze each statement and classify them into one of the following categories:\n\n- TP (True Positive): Statements present in the answer that are directly supported by one or more statements in the ground_truth,\n- FP (False Positive): Statements present in the answer that are not supported by any statements in the ground_truth,\n- FN (False Negative): Statements present in the ground_truth but absent in the answer.\n\nEach statement can only belong to one category: TP, FP, or FN. Provide the reason for your classification.",
        "output_format_instruction": "Output should be a well-formed JSON instance conforming to the following JSON schema.\n\nExample: For schema {\"properties\": {\"foo\": {\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {\"type\": \"string\"}}}, \"required\": [\"foo\"]}\n, the object {\"foo\": [\"bar\", \"baz\"]} is a valid schema instance. The object {\"properties\": {\"foo\": [\"bar\", \"baz\"]}} is invalid.\n\nOutput JSON schema:\n```\n{\"type\": \"object\", \"properties\": {\"TP\": {\"title\": \"Tp\", \"type\": \"array\", \"items\": {\"type\": \"object\"}}, \"FP\": {\"title\": \"Fp\", \"type\": \"array\", \"items\": {\"type\": \"object\"}}, \"FN\": {\"title\": \"Fn\", \"type\": \"array\", \"items\": {\"type\": \"object\"}}}, \"required\": [\"TP\", \"FP\", \"FN\"]}\n```\n\nReturn only the JSON string enclosed in triple backticks without any preamble or explanation.",
        "examples": [
            {
                "question": "What is the power source of the sun and what is its primary function?",
                "answer": ["The power source of the sun is nuclear fission, similar to nuclear reactors on Earth.", "The primary function of the sun is to provide light to the solar system."],
                "ground_truth": ["The power source of the sun is nuclear fusion, where hydrogen atoms fuse to form helium.", "This fusion process in the sun's core releases enormous amounts of energy.", "The sun's energy provides heat and light essential for life on Earth.", "Sunlight plays a critical role in Earth's climate system.", "Sunlight helps drive weather patterns and ocean currents."],
                "classification": {
                    "TP": [
                        {
                            "statement": "The primary function of the sun is to provide light to the solar system.",
                            "reason": "This statement is partially supported by the ground truth, which mentions sunlight and its roles, though it more broadly emphasizes the sun's energy."
                        }
                    ],
                    "FP": [
                        {
                            "statement": "The power source of the sun is nuclear fission, similar to nuclear reactors on Earth.",
                            "reason": "This statement is incorrect and contradicts the ground truth, which states the sun's power source is nuclear fusion."
                        }
                    ],
                    "FN": [
                        {
                            "statement": "The power source of the sun is nuclear fusion, where hydrogen atoms fuse to form helium.",
                            "reason": "This accurate description of the sun's power source is not included in the answer."
                        },
                        {
                            "statement": "This fusion process in the sun's core releases enormous amounts of energy.",
                            "reason": "The process and its significance are not mentioned in the answer."
                        },
                        {
                            "statement": "The sun's energy provides heat and light essential for life on Earth.",
                            "reason": "The answer mentions only light, omitting heat and its necessity for life covered in the ground truth."
                        },
                        {
                            "statement": "Sunlight plays a critical role in Earth's climate system.",
                            "reason": "The broader impact of sunlight on Earth's climate system is not mentioned in the answer."
                        },
                        {
                            "statement": "Sunlight helps drive weather patterns and ocean currents.",
                            "reason": "The impact of sunlight on weather patterns and ocean currents is missing from the answer."
                        }
                    ]
                }
            },
            {
                "question": "What is the boiling point of water?",
                "answer": ["The boiling point of water is 100 degrees Celsius at sea level."],
                "ground_truth": ["The boiling point of water is 100 degrees Celsius (212 degrees Fahrenheit) at sea level.", "The boiling point of water changes with altitude."],
                "classification": {
                    "TP": [
                        {
                            "statement": "The boiling point of water is 100 degrees Celsius at sea level.",
                            "reason": "This statement is directly supported by the ground truth, which explicitly states this fact."
                        }
                    ],
                    "FP": [],
                    "FN": [
                        {
                            "statement": "The boiling point of water changes with altitude.",
                            "reason": "The additional information about altitude affecting the boiling point is not mentioned in the answer."
                        }
                    ]
                }
            }
        ]
    }

