static_rubric = {
    "dimension_weight": {"readability": 0.25, "insight": 0.25, "comprehensiveness": 0.25, "instruction_following": 0.25},
    "criterions":
        {
            "comprehensiveness":
            [
                {
                    "criterion": "信息覆盖广度",
                    "explanation": "评估文章是否覆盖了与主题相关的所有关键领域和方面，不遗漏重要信息。",
                    "weight": 0.25
                },
                {
                    "criterion": "信息深度与细节",
                    "explanation": "评估文章是否提供了足够深入的细节信息，而不只是浅层次的概述。",
                    "weight": 0.25
                },
                {
                    "criterion": "数据与事实支持",
                    "explanation": "评估文章是否提供了充分的数据、事实、案例或证据来支持其论点和分析。",
                    "weight": 0.25
                },
                {
                    "criterion": "多角度与平衡性",
                    "explanation": "评估文章是否从多个角度考虑问题，并在相关情况下提供平衡的观点。",
                    "weight": 0.25
                },
            ],
        
            "insight":
            [
                {
                    "criterion": "分析深度与原创性",
                    "explanation": "评估文章是否提供了深入的分析和原创性的见解，而非简单重复已知信息。",
                    "weight": 0.25
                },
                {
                    "criterion": "逻辑推理与因果关系",
                    "explanation": "评估文章是否展示了清晰的逻辑推理，有效解释了现象背后的因果关系。",
                    "weight": 0.25
                },
                {
                    "criterion": "问题洞察与解决方案",
                    "explanation": "评估文章是否识别出关键问题或挑战，并提供了有见地的解决方案或建议。",
                    "weight": 0.25
                },
                {
                    "criterion": "前瞻性与启发性",
                    "explanation": "评估文章是否具有前瞻性思考，能够预见趋势并提供启发性的观点。",
                    "weight": 0.25
                }
            ],
            "instruction_following":
            [
                {
                    "criterion": "任务目标的回应",
                    "explanation": "评估文章是否直接回应了任务的核心目标和问题。",
                    "weight": 0.34
                },
                {
                    "criterion": "范围限定的遵守",
                    "explanation": "评估文章是否严格遵守了任务中设定的范围限定（如地域、时间、对象等）。",
                    "weight": 0.33
                },
                {
                    "criterion": "任务要求的完整覆盖",
                    "explanation": "评估文章是否完整覆盖了任务中提出的所有子问题或方面，没有遗漏重要部分。",
                    "weight": 0.33
                }
            ],
            "readability":
            [
                {
                    "criterion": "结构清晰与逻辑",
                    "explanation": "评估文章是否有清晰的结构，包括合适的引言、主体、结论，以及逻辑上连贯的段落组织。",
                    "weight": 0.25
                },
                {
                    "criterion": "语言表达与流畅性",
                    "explanation": "评估文章的语言是否清晰、准确、流畅，没有明显的语法错误或表达不当。",
                    "weight": 0.25
                },
                {
                    "criterion": "专业术语的恰当使用",
                    "explanation": "评估文章是否恰当使用专业术语，并在必要时提供解释，便于理解。",
                    "weight": 0.25
                },
                {
                    "criterion": "信息呈现与视觉效果",
                    "explanation": "评估文章是否有效使用格式、标题、列表、强调等手段来增强可读性，以及是否合理使用图表或其他视觉元素（如有）。",
                    "weight": 0.25
                }
            ]
    }
}
