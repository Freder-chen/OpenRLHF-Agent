from openrlhf_agent import AgentRuntime, OpenAIEngine, make_environment, make_template

if __name__ == "__main__":
    engine = OpenAIEngine(
        model="qwen3",
        base_url="http://localhost:8009/v1",
        api_key="empty",
    )
    env = make_environment(name="default")
    template = make_template(name="qwen3_instruct")

    rt = AgentRuntime(engine, env, template)
    # messages = [{"role": "user", "content": "Tell me a joke about programming."}]
    messages = [{"role": "user", "content": "思考，然后给我讲个笑话"}]
    for step in rt.run_steps(messages):
        print(step)
