## 文件夹结构

```text
deep_research_agent/
    ├── data/
    │   └── rl_train_q_rubrics.jsonl          # https://huggingface.co/datasets/rl-research/dr-tulu-rl-data 重构格式
    ├── prompt/                               # 提示词
    ├── reward/
    │   ├── api.py                            # 封装 LLM 异步调用
    │   ├── rubric.py                         # 静态通用标准
    │   └── report_reward/
    │       ├── report_reward.py              # 动态标准，报告对比打分
    │       ├── report_reward_no_ref.py       # 动态标准，单一报告打分
    │       └── report_reward_no_ref_rubrics.py  # 静态标准，单一报告打分
    ├── tool/
    │   ├── tool_monitor.py                   # 工具调用粗统计
    │   ├── my_tool.py                        # baidu_search, 360_crawler
    │   └── my_tool_test_time_v2.py           # 融合粗统计工具调用数据(baidu_search, 360_crawler, JinaReader)
    ├── agent_func.py                         # Agent 类定义
    └── train_reinforce_agent.sh              # 训练启动脚本
```
